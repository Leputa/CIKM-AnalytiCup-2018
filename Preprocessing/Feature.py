import sys
sys.path.append('../')

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Doc2Vec
import os
import numpy as np

from Preprocessing import Preprocess
from Config import config
from Model import Embeddings


class Feature():
    def __init__(self):
        self.preprocess = Preprocess.Preprocess()
        self.embeddings = Embeddings.Embeddings()

    def word_tf_idf(self, tag='word'):
        print("Tfidf on " + tag)

        if tag == 'word':
            path = config.cache_prefix_path + 'Tfidf_word.pkl'
        elif tag == 'char':
            path = config.cache_prefix_path + 'Tfidf_char.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        es = self.preprocess.load_all_data()[0]
        corpus = [" ".join(sentence) for sentence in es]

        if tag == 'word':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='word',
                ngram_range=(1, 3),
                max_features=10000
            )
        elif tag == 'char':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='char',
                ngram_range=(1, 4),
                max_features=30000
            )
        vectorizer.fit(corpus)

        # train
        _, _, train_left, train_right, train_labels = self.preprocess.load_train_data('en')
        train_data = [" ".join(train_left[i]) + " . " + " ".join(train_right[i]) for i in range(len(train_left))]
        train_features = vectorizer.transform(train_data)

        # dev
        _, _, dev_left, dev_right, dev_labels = self.preprocess.load_train_data('es')
        dev_data = [" ".join(dev_left[i]) + " . " + " ".join(dev_right[i]) for i in range(len(dev_left))]
        dev_features = vectorizer.transform(dev_data)

        # test
        test_left, test_right = self.preprocess.load_test()
        test_data = [" ".join(test_left[i]) + " . " + " ".join(test_right[i]) for i in range(len(test_left))]
        test_features = vectorizer.transform(test_data)

        with open(path, 'wb') as pkl:
            pickle.dump(((train_features, train_labels), (dev_features, dev_labels), test_features), pkl)

        return ((train_features, train_labels), (dev_features, dev_labels), test_features)


    def get_doc2vec(self, tag):
        print("getting doc2vec......")

        path = config.cache_prefix_path + 'doc2vec_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pickle) 

        model = Doc2Vec.load(config.cache_prefix_path + "doc2vec.embedding")

        # train
        _, _, train_left, train_right, train_labels = self.preprocess.load_train_data('en')
        train_left_vector = model.infer_vector(train_left)
        train_right_vector = model.infer_vector(train_right)
        train_vec = 

        # dev 
        _, _, dev_left, dev_right, dev_labels = self.preprocess.load_train_data('es')
        dev_left_vector = model.infer_vector(dev_left)
        dev_right_vector = model.infer_vector(dev_right)
        dev_vec = 

        # test
        test_left, test_right = self.preprocess.load_test()
        test_left_vector = model.infer_vector(test_left)
        test_right_vector = model.infer_vector(test_right)
        test_vec = 

        with open(path, 'wb') as pkl:
            pickle.dump(((train_vec, train_labels),(dev_vec, dev_labels), test_vec), pkl)

        return ((train_vec, train_labels),(dev_vec, dev_labels), test_vec)

    def get_average_word2vec(self):
        print ("getting average word2vec")

        path = config.cache_prefix_path + 'word2vec_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)


        embedding_matrix = self.embeddings.get_es_embedding_matrix()
        
        # train
        train_left, train_right, train_labels = self.preprocess.get_es_index_data('train')
        train_features = self.deal_average_word2vec(train_left, train_right, embedding_matrix)
        #dev
        dev_left, dev_right, dev_labels = self.preprocess.get_es_index_data('dev')
        dev_features = self.deal_average_word2vec(dev_left, dev_right, embedding_matrix)

        test_left, test_right = self.preprocess.get_es_index_data('test')
        test_features = self.deal_average_word2vec(test_left, test_right, embedding_matrix)

        with open(path, 'wb') as pkl:
            pickle.dump(((train_features, train_labels), (dev_features, dev_labels), test_features), pkl)

        return ((train_features, train_labels), (dev_features, dev_labels), test_features)


    def deal_average_word2vec(self, left_index, right_index, embedding_matrix):
        left_feature = np.ones((len(left_index), self.embeddings.vec_dim), dtype="float32") * 0.01
        right_feature = np.ones((len(right_index), self.embeddings.vec_dim), dtype="float32") * 0.01

        for i in range(len(left_index)):
            tmp_vec_left, tmp_vec_right = np.zeros(self.embeddings.vec_dim), np.zeros(self.embeddings.vec_dim)

            for index in left_index[i]:
                tmp_vec_left += embeddings_matrix[index]
            for index in right_index[i]:
                tmp_vec_right += embedding_matrix[index]

            tmp_vec_left /= len(left_index[i])
            tmp_vec_right /= len(right_index[i])
            
            left_feature[i] = tmp_vec_left
            right_feature[i] = tmp_vec_right

        return 



if __name__ == '__main__':
    feature = Feature()
    feature.word_tf_idf('char')
