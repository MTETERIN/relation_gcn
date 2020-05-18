"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction
Difference compared to MichSchli/RelationPrediction
* report raw metrics instead of filtered metrics
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data

from layers import RGCNBlockLayer as RGCNLayer
from model import BaseRGCN

import utils

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = self.embedding(node_id)

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=act, self_loop=True, dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g):
        return self.rgcn.forward(g)

    def evaluate(self, g):
        # get embedding and relation weight without grad
        embedding = self.forward(g)
        return embedding, self.w_relation

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        embedding = self.forward(g)
        score = self.calc_score(embedding, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embedding)
        return predict_loss + self.reg_param * reg_loss

def tsne(writers, readers, num=None):
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    model = torch.load('Writers&Readers.pt')
    entities = model._modules['rgcn']._modules['layers']._modules['0'].embedding.weight.detach().numpy()
    new_data_np = []
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    if num is None:
        relations = model.w_relation.detach().numpy()
        tsne_results = tsne.fit_transform(relations)
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(10, 10))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="tsne-2d-one",
            palette=sns.color_palette("hls", 641),
            data=df_subset,
            legend=False,
            alpha=1
        )
        fig = sns_plot.get_figure()
        fig.savefig("TSNE-641-relations.png")
    else:
        all_infofmation = writers[:20] + readers[:20]
        for data in all_infofmation:
            new_data_np.append(entities[data[1], :])
        tsne_results = tsne.fit_transform(new_data_np)
        colors = ['#008100'] * 20 + ['#0015ff'] * 20
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(10, 10))
        sns_plot = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="tsne-2d-one",
            palette=sns.color_palette(colors),
            data=df_subset,
            legend=False,
            alpha=1
        )
        for line in range(0, df_subset.shape[0]):
            sns_plot.text(df_subset['tsne-2d-one'][line] + 0.01, df_subset['tsne-2d-two'][line],
                          all_infofmation[line][0], horizontalalignment='left',
                          size='medium', color='black', weight='semibold')
        fig = sns_plot.get_figure()
        fig.savefig("TSNE-40.png")

def umap(writers, readers, num=None):
    import umap
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    model = torch.load('Writers&Readers.pt')
    entities = model._modules['rgcn']._modules['layers']._modules['0'].embedding.weight.detach().numpy()
    if num is None:
        relations = model.w_relation.detach().numpy()
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(relations)
        df_subset = pd.DataFrame()
        df_subset['x'] = embedding[:, 0]
        df_subset['y'] = embedding[:, 1]

        df_subset = pd.DataFrame()
        df_subset['umap-2d-one'] = embedding[:, 0]
        df_subset['umap-2d-two'] = embedding[:, 1]
        plt.figure(figsize=(16, 10))
        sns_plot = sns.scatterplot(
            x="umap-2d-one", y="umap-2d-two",
            hue="umap-2d-two",
            palette=sns.color_palette("hls", 641),
            data=df_subset,
            legend=False,
            alpha=0.3
        )
        fig = sns_plot.get_figure()
        fig.savefig("umap-relations-641.png")
    else:
        all_infofmation = writers[:20] + readers[:20]
        new_data_np = []
        for data in all_infofmation:
            new_data_np.append(entities[data[1], :])
        reducer = umap.UMAP()
        colors = ['#008100'] * 20 + ['#0015ff'] * 20
        embedding = reducer.fit_transform(new_data_np)
        df_subset = pd.DataFrame()
        df_subset['x'] = embedding[:, 0]
        df_subset['y'] = embedding[:, 1]

        df_subset = pd.DataFrame()
        df_subset['umap-2d-one'] = embedding[:, 0]
        df_subset['umap-2d-two'] = embedding[:, 1]
        plt.figure(figsize=(16, 10))
        sns_plot = sns.scatterplot(
            x="umap-2d-one", y="umap-2d-two",
            hue="umap-2d-two",
            palette=sns.color_palette(colors),
            data=df_subset,
            legend=False,
            alpha=0.3
        )
        for line in range(0, df_subset.shape[0]):
            sns_plot.text(df_subset['umap-2d-one'][line] + 0.01, df_subset['umap-2d-two'][line],
                          all_infofmation[line][0], horizontalalignment='left',
                          size='medium', color='black', weight='semibold')
        fig = sns_plot.get_figure()
        fig.savefig("umap-40.png")

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def _read_triplets(filename):
    with open(filename, 'r+', encoding='utf-8') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        try:
            s = entity_dict[triplet[0]]
            r = relation_dict[triplet[1]]
            o = entity_dict[triplet[2]]
            l.append([s, r, o])
        except BaseException as e:
            print(f"Error: {e}")
    return l

def training_process(num_nodes, train_data, valid_data, test_data, num_rels):
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = torch.load('Writers&Readers.pt')
    # model = LinkPredict(num_nodes,
    #                     args.n_hidden,
    #                     num_rels,
    #                     num_bases=args.n_bases,
    #                     num_hidden_layers=args.n_layers,
    #                     dropout=args.dropout,
    #                     use_cuda=use_cuda,
    #                     reg_param=args.regularization)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
        range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)
    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
    test_graph.edata['type'] = test_rel

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample)
        print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        node_norm = torch.from_numpy(node_norm).view(-1, 1)
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, node_norm = edge_type.cuda(), node_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
        g.ndata.update({'id': node_id, 'norm': node_norm})
        g.edata['type'] = edge_type

        t0 = time.time()
        loss = model.get_loss(g, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            torch.save(model, 'Writers&Readers.pt')
            model.eval()
            print("start eval")
            mrr = utils.evaluate(test_graph, model, valid_data,
                                 hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)
            if use_cuda:
                model.cuda()
            print("training done")
            print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
            print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

            print("\nstart testing:")
            # use best model checkpoint
            checkpoint = torch.load(model_state_file)
            if use_cuda:
                model.cpu()  # test on CPU
            model.eval()
            model.load_state_dict(checkpoint['state_dict'])
            print("Using best epoch: {}".format(checkpoint['epoch']))
            utils.evaluate(test_graph, model, test_data, hits=[1, 3, 10],
                           eval_bz=args.eval_batch_size)

def find_entity_by_index(index):
    with open('data/entities.dict', 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] == str(index):
                return line[1]

def closest(writers, readers, type='readers'):
    import os
    model = torch.load('Writers&Readers.pt')
    if type == 'writers':
        entities = model._modules['rgcn']._modules['layers']._modules['0'].embedding.weight.detach().numpy()
        all_infofmation = writers[:2] + readers[:3]
        all_writers_readers = writers + readers
        new_data_np = []
        for data in all_writers_readers:
            new_data_np.append(entities[data[1], :])
        from scipy import spatial
        tree = spatial.KDTree(new_data_np)
        for relation in all_infofmation:
            results = tree.query(entities[relation[1]], k=6)
            print(tree.query(entities[relation[1]], k=6))
            elements = results[1]
            for result in elements:
                print(f'Top closest for {relation[0]}:  {find_entity_by_index(result)}')
    else:
        relations = model.w_relation.detach().numpy()
        top_5_relations = get_all_relations()[:5]
        from scipy import spatial
        tree = spatial.KDTree(relations)
        for relation in top_5_relations:
            results = tree.query(relations[relation[1]], k=6)
            print(results)
            elements = results[1]
            for result in elements:
                print(f'Top closest for {relation[0]}:  {get_all_relations()[result]}')


def getEmbeddingIndex(name):
    with open('data/entities.dict', 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[1] == name:
                return int(line[0])
    print('We can"t find index for:' % name)


def get_all_relations():
    relations = []
    with open('data/relations.dict', 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            relations.append((line[1], int(line[0])))
    return relations


def get_all_writers_readers():
    writers = []
    readers = []
    with open('data/writers_and_books.tsv', 'r+', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            writer_index = getEmbeddingIndex(line[0])
            writers.append((line[0], writer_index))
            reader_index = getEmbeddingIndex(line[1])
            readers.append((line[1], reader_index))
    return writers, readers


def main(args):

    # load graph data
    writers, readers = get_all_writers_readers()
    if args.tsne:
        if args.tsne == 40:
            tsne(writers, readers, 40)
        else:
            tsne(writers, readers)
    elif args.closest == 'writers' or args.closest == 'relations':
        closest(writers, readers, args.data)
    elif args.umap:
        if args.umap == 40:
            umap(writers, readers, 40)
        else:
            umap(writers, readers)
    elif args.data:
        import os
        entity_path = os.path.join('data', 'entities.dict')
        relation_path = os.path.join('data', 'relations.dict')
        train_path = os.path.join('data', 'train.txt')
        valid_path = os.path.join('data', 'valid.txt')
        test_path = os.path.join('data', 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        num_nodes = len(entity_dict)
        print("# entities: {}".format(num_nodes))
        num_rels = len(relation_dict)
        print("# relations: {}".format(num_rels))
        print("# edges: {}".format(len(train)))
        num_nodes = num_nodes
        train_data = train
        valid_data = valid
        test_data = test
        num_rels = num_rels
        training_process(num_nodes, train_data, valid_data, test_data, num_rels)
    else:
        dataset = 'FB15k-237'
        data = load_data(dataset)
        num_nodes = data.num_nodes
        train_data = data.train
        valid_data = data.valid
        test_data = data.test
        num_rels = data.num_rels
        training_process(num_nodes, train_data, valid_data, test_data, num_rels)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--tsne", type=int, default=0,
                        help="tsne")
    parser.add_argument("--umap", type=int, default=0,
                        help="umap")
    parser.add_argument("--closest", type=str, default='1',
                        help="closest")
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=100,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=50,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--data", type=int, default=0,
                        help="Use our data or training data")

    args = parser.parse_args()
    print(args)
    main(args)