import sys
sys.path.append('../')

from Preprocessing import Preprocess
from Preprocessing import Feature
from Config import config
from Config.utils import MathUtil

import networkx as nx
import os
import pickle
from tqdm import tqdm
import numpy as np

class GraphFeature():
    def __init__(self):
        self.preprocess = Preprocess.Preprocess()
        self.Feature = Feature.Feature()

    def load_left_right(self, tag):
        dic = {
            'train':'en',
            'dev': 'es'
        }
        labels = None
        if tag == 'train' or tag == 'dev':
            _, _, left, right, labels = self.preprocess.load_train_data(dic[tag])
        elif tag == 'test':
            left, right = self.preprocess.load_test()
        return left, right, labels

    def sentence2id(self):
        def add_s2id(tag, s2id):
            left, right = self.load_left_right(tag)
            for i in range(len(left)):
                left_sentence = " ".join(left[i])
                right_sentence = " ".join(right[i])
                if left_sentence not in s2id:
                    s2id[left_sentence] = len(s2id)
                if right_sentence not in s2id:
                    s2id[right_sentence] = len(s2id)
            return s2id

        print('getting sentence to id...')
        path = config.cache_prefix_path + 'sentence2id.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        s2id = {}
        s2id = add_s2id('train', s2id)
        s2id = add_s2id('dev', s2id)
        s2id = add_s2id('test', s2id)

        with open(path, 'wb') as pkl:
            pickle.dump(s2id, pkl)
        return s2id

    def get_sentence_index_data(self, tag):
        print("getting sentence index data...")
        if tag == 'train':
            path = config.cache_prefix_path + 'sentence2index_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'sentence2index_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'sentence2index_test.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        s2id = self.sentence2id()
        left, right, labels = self.load_left_right(tag)
        left_ids, right_ids = [], []

        for i in range(len(left)):
            s_left = ' '.join(left[i])
            s_right = ' '.join(right[i])
            left_ids.append(s2id[s_left])
            right_ids.append(s2id[s_right])

        with open(path, 'wb') as pkl:
            pickle.dump((left_ids, right_ids, labels), pkl)

        return (left_ids, right_ids, labels)


    def generate_graph(self, feature_tag):
        def add_G(tag, feature_tag ,G, e2weight):
            left_ids, right_ids, labels = self.get_sentence_index_data(tag)

            if feature_tag == 'lsa_sim':
                weights  = self.Feature.get_lsa_sim(tag)
            elif feature_tag == 'char_sim':
                weights = self.Feature.get_tfidf_sim(tag, 'char')
            elif feature_tag == 'label':
                if tag == 'train' or tag == 'dev':
                    # 这样会在dev时泄漏数据，感觉不靠谱
                    weights = labels
                else:
                    weights = [0.5]*len(left_ids)
            elif feature_tag == 'label_2':
                if tag == 'train' or tag == 'dev':
                    for i in range(len(left_ids)):
                        if labels[i] == 1:
                            G.add_edge(left_ids[i], right_ids[i], weights=labels[i])
                            e2weight[(left_ids[i], right_ids[i])] = 1
                            e2weight[(right_ids[i], left_ids[i])] = 1
                        else:
                            G.add_edge(left_ids[i], right_ids[i], weights=100)
                            e2weight[(left_ids[i], right_ids[i])] = 100
                            e2weight[(right_ids[i], left_ids[i])] = 100
                elif tag == 'test':
                    for i in range(len(left_ids)):
                        G.add_node(left_ids[i])
                        G.add_node(right_ids[i])
                return G, e2weight

            for i in range(len(left_ids)):
                if weights[i] > 1:
                    weights[i] = 1
                G.add_edge(left_ids[i], right_ids[i], weight = 1 - weights[i])
                e2weight[(left_ids[i], right_ids[i])] = 1 - weights[i]
                e2weight[(right_ids[i], left_ids[i])] = 1 - weights[i]
            return G, e2weight

        print('generating graph...')
        path = config.cache_prefix_path + feature_tag +'_graph.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        G = nx.Graph()
        e2weight = {}
        G, e2weight = add_G('test', feature_tag, G, e2weight)
        G, e2weight = add_G('train', feature_tag, G, e2weight)
        G, e2weight = add_G('dev', feature_tag, G, e2weight)

        with open(path, 'wb') as pkl:
            pickle.dump((G, e2weight), pkl)

        return G, e2weight

    def get_shortest_path(self, tag, feature_tag):
        def extract_row(qid1, qid2, G, e2weight, feature_tag):
            if feature_tag == 'label_2':
                shortest_path = -1
            else:
                shortest_path = 100
            if tag =='test':
                if nx.has_path(G, qid1, qid2):
                    shortest_path =  nx.dijkstra_path_length(G, qid1, qid2)
            else:
                G.remove_edge(qid1, qid2)
                if nx.has_path(G, qid1, qid2):
                    shortest_path =  nx.dijkstra_path_length(G, qid1, qid2)
                G.add_edge(qid1, qid2, weight = e2weight[(qid1, qid2)])

            return [shortest_path]

        print('getting shortest path...')
        path = config.cache_prefix_path + tag + '_' + feature_tag + '_shortest_path.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left_ids, right_ids, _ = self.get_sentence_index_data(tag)
        G, e2weight = self.generate_graph(feature_tag)

        feature = []
        for i in tqdm(range(len(left_ids))):
            feature.append(extract_row(left_ids[i], right_ids[i], G, e2weight, feature_tag))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def generate_pagerank(self, feature_tag, alpha, max_iter):
        path = config.cache_prefix_path + feature_tag + '_pagerank.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        G, e2weight = self.generate_graph(feature_tag)
        pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter)

        with open(path, 'wb') as pkl:
            pickle.dump(pr, pkl)
        return pr

    def generate_hits(self, feature_tag, max_iter):
        path = config.cache_prefix_path + feature_tag + '_hits.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        G, e2weight = self.generate_graph(feature_tag)
        hits_h, hits_a = nx.hits(G, max_iter=max_iter, tol=1e-05)

        with open(path, 'wb') as pkl:
            pickle.dump((hits_h, hits_a), pkl)
        return hits_h, hits_a

    def generate_graph_clique(self, feature_tag):
        # 最大团
        G, _ = self.generate_graph(feature_tag)
        n2clique = {} #结点：该节点所处的各个最大团的ID
        cliques = []
        for clique in nx.find_cliques(G):
            for n in clique:
                if n not in n2clique:
                    n2clique[n] = []
                n2clique[n].append(len(cliques))
            cliques.append(clique)
        return n2clique, cliques

    def get_pagerank(self, tag, feature_tag):
        def extract_row(qid1, qid2, pagerank):
            pr_left = pagerank[qid1] * 1e3
            pr_right = pagerank[qid2] * 1e3

            return [pr_left, pr_right, max(pr_left, pr_right), (pr_left + pr_right)/2]

        print('getting pagerank feature...')
        path = config.cache_prefix_path + tag + '_' + feature_tag + '_pagerank_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left_ids, right_ids, _ = self.get_sentence_index_data(tag)
        pr = self.generate_pagerank(feature_tag, alpha=0.5, max_iter=200)

        feature = []
        for i in tqdm(range(len(left_ids))):
            feature.append(extract_row(left_ids[i], right_ids[i], pr))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_hits(self, tag, feature_tag):
        # 这个特征大概没用
        def extract_row(qid1, qid2, hits_h, hits_a):
            h1 = hits_h[qid1] * 1e3
            h2 = hits_h[qid2] * 1e3
            a1 = hits_a[qid1] * 1e3
            a2 = hits_a[qid2] * 1e3
            return [h1, h2, a1, a2,
                    max(h1, h2), max(a1, a2),
                    min(h1, h2), min(a1, a2),
                    (h1 + h2) / 2., (a1 + a2) / 2.]

        print('getting hits feature...')
        path = config.cache_prefix_path + tag + '_' + feature_tag + '_hits_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left_ids, right_ids, _ = self.get_sentence_index_data(tag)
        hits_h, hits_a = self.generate_hits(feature_tag, max_iter=5000)

        feature = []
        for i in tqdm(range(len(left_ids))):
            feature.append(extract_row(left_ids[i], right_ids[i], hits_h, hits_a))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_max_cliques_size(self, tag, feature_tag):
        # 这个特征也没啥用
        def extract_row(qid1, qid2, n2clique, cliques):
            max_clique_size = 0
            for clique_id in n2clique[qid1]:
                if qid2 in cliques[clique_id]:
                    max_clique_size = max(max_clique_size, len(cliques[clique_id]))
            return [max_clique_size]

        print("getting max cliques size...")
        path = config.cache_prefix_path + tag + '_' + feature_tag + '_max_clique_size_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left_ids, right_ids, _ = self.get_sentence_index_data(tag)
        n2clique, cliques = self.generate_graph_clique(feature_tag)

        feature = []
        for i in tqdm(range(len(left_ids))):
            feature.append(extract_row(left_ids[i], right_ids[i], n2clique, cliques))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_share_num_clique(self, tag, feature_tag):
        # 这个特征居然都没啥用
        def extract_row(qid1, qid2, n2clique, cliques):
            num_clique = 0
            for clique_id in  n2clique[qid1]:
                if qid2 in cliques[clique_id]:
                    num_clique += 1
            return [num_clique]

        print('getting number of share cliques...')
        path = config.cache_prefix_path + tag + '_' + feature_tag + '_share_num_clique_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left_ids, right_ids, _ = self.get_sentence_index_data(tag)
        n2clique, cliques = self.generate_graph_clique(feature_tag)

        feature = []
        for i in tqdm(range(len(left_ids))):
            feature.append(extract_row(left_ids[i], right_ids[i], n2clique, cliques))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_neighbor_share_num(self, tag, feature_tag):
        # 这个特征居然也没用

        def extract_row(qid1, qid2, G, e2weight):
            left_neighbor = list(G.neighbors(qid1))
            right_neighbor = list(G.neighbors(qid2))


            inter = len(set(left_neighbor)&set(right_neighbor))
            outer = len(set(left_neighbor)|set(right_neighbor))

            IOU = inter/outer
            return [inter, IOU]

        print('getting neighbor share num...')
        path = config.cache_prefix_path + tag + '_' + feature_tag + '_neighbor_share_num_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left_ids, right_ids, _ = self.get_sentence_index_data(tag)
        G, e2weight = self.generate_graph(feature_tag)

        feature = []
        for i in tqdm(range(len(left_ids))):
            feature.append(extract_row(left_ids[i], right_ids[i], G, e2weight))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def add_addtional_feature(self, tag, feature_tag):
        short_path_1 = self.get_shortest_path(tag, feature_tag)
        pagerank_1 = self.get_pagerank(tag, feature_tag)
        pagerank_1 = pagerank_1.reshape((pagerank_1.shape[0], pagerank_1.shape[1]))

        return np.hstack([short_path_1, pagerank_1])


if __name__ == "__main__":
    graph = GraphFeature()
    graph.get_shortest_path('test', 'label')





