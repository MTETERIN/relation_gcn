from functools import partial
from collections import OrderedDict
from urllib.parse import urljoin
import re
import json
from numas.patch.pomp import AioHttpRequest
from ..pompaggregator import PompAggregatorNew, PompSpider
from ..entity import City, Item, Row, RedisRow
import re
from ..utils import getTransliteration

class CitySpider(PompSpider):
    BASE_URL = 'http://zhivika.ru/'
    PHARMACY_NAME = 'Аптека Живика'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ENTRY_REQUESTS = AioHttpRequest(self.BASE_URL, callback=self.searchCities)
        self.downloadDelay = 2
        self.connectionsLimit = 20
        self.taskCount = 10
        self.connectionsLimitPerHost = 20
        self.maxRequestErrors = 15  # MTETERIN 12112018 Need more mistakes. 3 too little.
        self.totalRequestNum = 1
        self.completedRequestsNum = 0

    def searchCities(self, response):
        """
                Looks for a list of cities at website and send search request about drugs
                :param response:
                :return:
        """
        self.completedRequestsNum += 1
        cities = response.transform(self.xsl('cities'), self.xpath('cities'))

        citiesCount = len(cities)
        itemCount = len(self.config['items'])
        self.totalRequestNum += citiesCount * itemCount

        if not cities:
            raise CitiesNotFoundError("Cities not found")

        for city in cities:
            request = OrderedDict()
            request['url'] = city.get('href')
            request['cookies'] = city.get('id')
            request['redisPostfix'] = f"{self.BASE_URL}:{getTransliteration(city.get('name'))}"
            yield RedisRow(request=request)

class Spider(PompSpider):
    """Class allows you to read drugs and pharmacies from http://zhivika.ru/
            @author: MTETERIN
            @Date: 18.12.2018
    """

    BASE_URL = 'http://zhivika.ru/'
    ITEMS_URL = "plugins/catalog/catalog_search"

    PHARMACY_NAME = 'Аптека Живика'
    VENDOR_NAME = 'ВЕРТЕКС'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ENTRY_REQUESTS = AioHttpRequest(self.BASE_URL, callback=self.extractCity)
        self.downloadDelay = 2
        self.connectionsLimit = 20
        self.taskCount = 10
        self.connectionsLimitPerHost = 20
        self.maxRequestErrors = 15  # MTETERIN 12112018 Need more mistakes. 3 too little.
        self.totalRequestNum = 1
        self.completedRequestsNum = 0

    def extractCity(self, response):
        """
                Looks for a list of cities at website and send search request about drugs
                :param response:
                :return:
        """
        self.completedRequestsNum += 1
        url = self.url('start')
        cookies = self.cookies('start')
        city = City(name=url.split('=')[1])
        yield from self.requestCityItems(
            City(
                id=cookies,
                name=city.name
            )
        )

    def requestCityItems(self, city):
        """
                Requests drugs for city
                :param city:
                :return:
        """
        items = self.config['items']
        itemCount = len(items)
        self.logger.debug(f"Start {itemCount} item requests for '{city.name}'")
        for searchItem in items:
            callback = partial(self.extractItems, city, searchItem, 1)
            yield AioHttpRequest(
                urljoin(self.url('start'), self.ITEMS_URL),
                data={
                    'searchText': searchItem
                },
                cookies={'current_city_name': '{c.id}'.format(c=city)},
                callback=callback
            )

    def getPath(self, path):

        from pathlib import Path
        if isinstance(path, Path):
            return path
        else:
            return Path(path)


    def extractItems(self, city, searchItem, page, response):
        """
                Get information about drugs
                :param city:
                :param searchItem:
                :param page:
                :param response:
                :return:
        """
        html_str = response.raw.decode(encoding='cp1251')
        import uuid
        new_uuid = str(uuid.uuid4())
        htmlFileName = f'{new_uuid}.txt'
        importPath = 'C:\\Users\\Mteterin\\Temp\\DataImport\\Import'

        filePath = self.getPath(importPath) / htmlFileName
        htmlFile = open(filePath, "w")
        htmlFile.write(html_str)
        htmlFile.close()

        isMissing = 'НЕТ'
        pages = response.transform(self.xsl('pages'), self.xpath('pages'))
        if pages:
            maxPage = int(max([current_page.get('page') for current_page in pages]))
        else:
            maxPage = 1

        self.completedRequestsNum += 1
        items = response.transform(self.xsl('items'), self.xpath('items'))
        real_items = []

        for item in items:
            itemName = item.get('name')
            real_items.append(itemName.strip())
        try:
            df2 = {'name': htmlFileName,
                       'items': str(real_items)}
            import pandas as pd
            df2 = pd.DataFrame.from_dict([df2])
            df = pd.read_csv('csv_example.csv', encoding='cp1251')
            df = pd.concat([df, df2], axis=0)
            df.to_csv('csv_example.csv', encoding='cp1251', index=False)
        except BaseException as e:
            print(e)
        if maxPage > 1 and page == 1:
            for page_number in range(2, maxPage + 1):
                self.logger.debug(f"Start '{searchItem}' request for '{city.name} at page #{page_number}'")
                self.totalRequestNum += 1

                callback = partial(self.extractItems, city, searchItem, page_number)

                yield AioHttpRequest(
                    response.data.base,
                    data={
                        'searchText': searchItem,
                        'pg': page_number
                    },
                    cookies=response.request.cookies,
                    callback=callback)

    def extractPharmacies(self, city, item, response):
        """
                Extract all information about drugs in
                :param city:
                :param item:
                :param response:
                :return:
        """
        answer = response.raw.decode(encoding='cp1251')
        self.completedRequestsNum += 1
        producersPattern = re.compile('(window.product_page_data = (.*?) <\/script>)', re.S)
        pharmacies_str = producersPattern.findall(answer)
        if pharmacies_str:
            pharmacies_dict = pharmacies_str[0][1]
            if not pharmacies_dict:
                self.logger.debug(f"Request № {self.completedRequestsNum}/{self.totalRequestNum}. We don't get any data for '{item.name}' in '{city} for some reason")
            else:
                pharmacies_json = json.loads(pharmacies_dict)
                pharmacies = pharmacies_json.get('ost_by_apt')
                self.logger.debug(
                    "Request № {completed}/{total}. Found {len} pharmacies for '{item.name}' in '{city}'".format(
                        completed=self.completedRequestsNum,
                        total=self.totalRequestNum,
                        len=len(pharmacies),
                        item=item,
                        city=city.name))
                for pharmacy in pharmacies:
                    if self.VENDOR_NAME not in pharmacy['Producer'].upper():
                        self.logger.debug(
                            "Request № {completed}/{total}. Vendor of item '{item}' is not 'Vertex'. It is '{brand}'".format(
                                completed=self.completedRequestsNum,
                                total=self.totalRequestNum,
                                item=item.name,
                                brand=pharmacy.get('Producer')
                            )
                        )
                        continue

                    if pharmacy.get('Addr'):
                        address = pharmacy.get('Addr').strip()
                    for prices in pharmacy['OstByPriceDg']:
                        price = prices.get('price')

                        if not address:
                            raise ValueError("Can't find address in '%s' string" % address)

                        if price:
                            yield Row(
                                item.name,
                                vendor=self.VENDOR_NAME,
                                city=city.name,
                                pharmacy=self.PHARMACY_NAME,
                                address=address,
                                price=str(price),
                            )


# MTETERIN 18122018
class CitiesNotFoundError(Exception):
    pass


class CityPlugin(PompAggregatorNew):
    AGGREGATOR_NAME = 'Аптека Живика'
    SPIDER = CitySpider


class Plugin(PompAggregatorNew):
    AGGREGATOR_NAME = 'Аптека Живика'
    SPIDER = Spider
