import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import Feature

import xgboost as xgb
from scipy.sparse import hstack
from scipy.sparse import vstack
import gc


class Xgboost():

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.Feature = Feature.Feature()
        self.params = {  'booster':'gbtree',
                         'num_leaves':32,
                         'max_depth':6,
                         'eta':0.05,
                         'max_bin':425,
                         'subsample_for_bin':50000,
                         'objective':'binary:logistic',
                         'min_split_gain':0,
                         'min_child_weight':6,
                         'min_child_samples':10,
                         'subsample':0.7,
                         'colsample_bytree':0.8,
                         'lambda':10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                         'alpha':1,   #L1正则化
                         'seed':2018,
                         'nthread':7,
                         'silent':True,
                         'gamma':0.1,
                         'eval_metric':'auc'
                    }
        self.num_rounds = 5000
        self.early_stop_rounds = 100

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

    def train(self, tag):
        print("Xgboost training")

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


        xgb_train = xgb.DMatrix(train_data, label=train_labels)
        xgb_val = xgb.DMatrix(dev_data, label=dev_labels)
        watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

        model = xgb.train(self.params, xgb_train, self.num_rounds, watchlist, early_stopping_rounds=self.early_stop_rounds)

    def test(self, name):
        print("Xgboost testing...")
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



        gc.collect()

        xgb_train = xgb.DMatrix(train_data, label=train_labels)
        xgb_test = xgb.DMatrix(test_data)

        num_rounds = 700
        watchlist = [(xgb_train, 'train')]
        model = xgb.train(self.params, xgb_train, num_rounds, watchlist)

        submit = model.predict(xgb_test)
        with open(config.output_prefix_path + name +'-summit.txt', 'w') as fr:
            for sub in submit:
                fr.write(str(sub) + '\n')


if __name__ == "__main__":
    model = Xgboost()
    model.train('doc2vec')
    #model.test(name='tfidf')


