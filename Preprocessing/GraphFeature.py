import sys
sys.path.append('../')

from Preprocessing import Preprocess
from Config import config
from Config.utils import MathUtil

import networkx as nx
import os
import pickle

class GraphFeature():
    def __init__(self):
        self.preprocess = Preprocess.Preprocess()

    def load_left_right(self, tag):
        dic = {
            'train':'en',
            'dev': 'es'
        }
        if tag == 'train' or tag == 'dev':
            _, _, left, right, _ = self.preprocess.load_train_data(dic[tag])
        elif tag == 'test':
            left, right = self.preprocess.load_test()

        return left, right

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
        left, right = self.load_left_right(tag)
        left_ids, right_ids = [], []

        for i in range(len(left)):
            s_left = ' '.join(left[i])
            s_right = ' '.join(right[i])
            left_ids.append(s2id[s_left])
            right_ids.append(s2id[s_right])

        with open(path, 'wb') as pkl:
            pickle.dump((left_ids, right_ids), pkl)

        return (left_ids, right_ids)


if __name__ == "__main__":
    graph = GraphFeature()
    #graph.sentence2id()
    graph.get_sentence_index_data('train')



