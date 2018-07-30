import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import Feature
from Preprocessing import PowerfulWord
from Preprocessing import GraphFeature

from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
import gc


class BaseMlModel():

    def __init__(self):

        self.preprocessor = Preprocess.Preprocess()
        self.Feature = Feature.Feature()
        self.Powerfulwords = PowerfulWord.PowerfulWord()
        self.Graph = GraphFeature.GraphFeature()

    def get_dov2vec_data(self):
        train_data, dev_data, test_data = self.Feature.get_doc2vec()
        _, _, train_labels = self.preprocessor.get_es_index_data('train')
        _, _, dev_labels = self.preprocessor.get_es_index_data('dev')
        return train_data, train_labels, dev_data, dev_labels, test_data

    def get_word2vec_data(self):
        (train_data, train_labels), (dev_data, dev_labels), _ = self.Feature.get_average_word2vec()
        return train_data, train_labels, dev_data, dev_labels, _

    def get_tfidf(self, tag='word'):
        (train_features, train_labels), (dev_features, dev_labels), test_features = self.Feature.get_tf_idf(tag)
        return train_features, train_labels, dev_features, dev_labels, test_features

    def get_lsa(self, tag='word'):
        (train_features, train_labels), (dev_features, dev_labels), test_features = self.Feature.LSA(tag)
        return train_features, train_labels, dev_features, dev_labels, test_features


    def prepare_train_data(self, tag, modeltype):

        if tag == 'tfidf':
            char_train_data, train_labels, char_dev_data, dev_labels, _ = self.get_tfidf('char')
            word_train_data, _, word_dev_data, _, _ = self.get_tfidf('word')
            train_data = hstack([char_train_data, word_train_data])
            dev_data = hstack([char_dev_data, word_dev_data])
        elif tag == 'tfidf_word':
            train_data, train_labels, dev_data, dev_labels, _ = self.get_tfidf('word')

        elif tag == 'tfidf_char':
            train_data, train_labels, dev_data, dev_labels, _ = self.get_tfidf('char')

        elif tag == 'lsa_word':
            train_data, train_labels, dev_data, dev_labels, _ = self.get_lsa('word')

        elif tag == 'word2vec':
            train_data, train_labels, dev_data, dev_labels, _ = self.get_word2vec_data()

        elif tag == 'doc2vec':
            train_data, train_labels, dev_data, dev_labels, _ = self.get_dov2vec_data()

        elif tag == 'concat_feature':
            char_train_data, train_labels, char_dev_data, dev_labels, _ = self.get_tfidf('char')
            word_train_data, _, word_dev_data, _, _ = self.get_tfidf('word')
            tfidf_train = hstack([char_train_data, word_train_data])
            tfidf_dev = hstack([char_dev_data, word_dev_data])

            word2vec_train, _, word2vec_dev, _, _ = self.get_word2vec_data() #加了这个后反而下降了
            word2vec_train = coo_matrix(word2vec_train)
            word2vec_dev = coo_matrix(word2vec_dev)

            train_data = hstack([tfidf_train, word2vec_train])
            dev_data = hstack([tfidf_dev, word2vec_dev])

        elif tag == 'human_feature':
            char_train_data, train_labels, char_dev_data, dev_labels, _ = self.get_tfidf('char')
            word_train_data, _, word_dev_data, _, _ = self.get_tfidf('word')
            train_data = hstack([char_train_data, word_train_data])
            dev_data = hstack([char_dev_data, word_dev_data])

            train_feature = coo_matrix(self.Feature.addtional_feature('train', modeltype))
            dev_feature = coo_matrix(self.Feature.addtional_feature('dev', modeltype))

            words_train_feature = coo_matrix(self.Powerfulwords.addtional_feature('train', modeltype))
            words_dev_features = coo_matrix(self.Powerfulwords.addtional_feature('dev', modeltype))

            # graph_train_feature = coo_matrix(self.Graph.add_addtional_feature('train', 'char_sim'))
            # graph_dev_feature =coo_matrix(self.Graph.add_addtional_feature('dev', 'char_sim'))

            train_data = hstack([char_train_data, word_train_data, train_feature, words_train_feature])
            dev_data = hstack([char_dev_data, word_dev_data, dev_feature, words_dev_features])

        return train_data, train_labels, dev_data, dev_labels


    def prepare_test_data(self, name, modeltype):
        if name == 'tfidf_word':
            train_data, train_labels, dev_data, dev_labels, test_data = self.get_tfidf('word')
            train_data = vstack([train_data, dev_data]).tocsr()

            train_labels.extend(dev_labels)

            del dev_data, dev_labels

        if name == 'tfidf_char':
            train_data, train_labels, dev_data, dev_labels, test_data = self.get_tfidf('char')
            train_data = vstack([train_data, dev_data]).tocsr()

            train_labels.extend(dev_labels)

            del dev_data, dev_labels

        elif name == 'tfidf':
            char_train_data, train_labels, char_dev_data, dev_labels, char_test_data = self.get_tfidf('char')
            word_train_data, _, word_dev_data, _, word_test_data = self.get_tfidf('word')

            train_data = vstack([hstack([char_train_data, word_train_data]), hstack([char_dev_data, word_dev_data])]).tocsr()
            train_labels.extend(dev_labels)


            test_data = hstack([char_test_data, word_test_data])
            del char_train_data, char_dev_data, dev_labels, char_test_data, word_train_data, word_test_data, word_dev_data

        elif name == 'lsa_word':
            train_data, train_labels, dev_data, dev_labels, test_data = self.get_lsa('word')
            train_data = vstack([train_data, dev_data]).tocsr()

            train_labels.extend(dev_labels)

            del dev_data, dev_labels

        elif name == 'human_feature':
            char_train_data, train_labels, char_dev_data, dev_labels, char_test_data = self.get_tfidf('char')
            word_train_data, _, word_dev_data, _, word_test_data = self.get_tfidf('word')

            train_data = vstack([hstack([char_train_data, word_train_data]), hstack([char_dev_data, word_dev_data])]).tocsr()
            train_labels.extend(dev_labels)

            train_feature = coo_matrix(self.Feature.addtional_feature('train', modeltype))
            dev_feature = coo_matrix(self.Feature.addtional_feature('dev', modeltype))

            train_feature = vstack([train_feature, dev_feature])
            test_feature = coo_matrix(self.Feature.addtional_feature('test_b', modeltype))


            words_train_feature = coo_matrix(self.Powerfulwords.addtional_feature('train', modeltype))
            words_dev_feature = coo_matrix(self.Powerfulwords.addtional_feature('dev', modeltype))

            words_train_feature = vstack([words_train_feature, words_dev_feature])
            words_test_feature = coo_matrix(self.Powerfulwords.addtional_feature('test_b', modeltype))

            # graph_train_feature = coo_matrix(self.Graph.add_addtional_feature('train', 'char_sim'))
            # graph_dev_feature = coo_matrix(self.Graph.add_addtional_feature('dev', 'char_sim'))
            #
            # graph_train_feature = vstack([graph_train_feature.astype('float'), graph_dev_feature.astype('float')])
            # graph_test_feature = coo_matrix(self.Graph.add_addtional_feature('test', 'char_sim')).astype('float')

            train_data = hstack([train_data, train_feature, words_train_feature])
            test_data = hstack([char_test_data, word_test_data, test_feature, words_test_feature])

            del char_train_data, char_dev_data, dev_labels, char_test_data, word_train_data, word_test_data, word_dev_data


        return train_data, train_labels, test_data



if __name__ == "__main__":
    model = Xgboost()
    #model.train('tfidf')
    model.test(name='tfidf')