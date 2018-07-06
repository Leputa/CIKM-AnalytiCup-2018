import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import Feature
from Preprocessing import PowerfulWord
from Model.BaseMlModel import BaseMlModel

import xgboost as xgb



class Xgboost(BaseMlModel):

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.Feature = Feature.Feature()
        self.Powerfulwords = PowerfulWord.PowerfulWord()

        self.params = {  'booster':'gbtree',
                         'max_depth':6,
                         'eta':0.05,
                         # 'max_bin':425,
                         # 'subsample_for_bin':50000,
                         'objective':'binary:logistic',
                         # 'min_split_gain':0,
                         # 'min_child_weight':6,
                         # 'min_child_samples':10,
                         'subsample':0.7,
                         'colsample_bytree':0.7,
                         'lambda':10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                         'alpha':1,    # L1正则化
                         'seed':2018,
                         'nthread':7,
                         'silent':True,
                         'gamma':0.1,
                         'eval_metric':'auc'
                    }
        self.num_rounds = 5000
        self.early_stop_rounds = 200


    def train(self, tag):
        print("Xgboost training")

        train_data, train_labels, dev_data, dev_labels = self.prepare_train_data(tag)

        xgb_train = xgb.DMatrix(train_data, label=train_labels)
        xgb_val = xgb.DMatrix(dev_data, label=dev_labels)
        watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

        model = xgb.train(self.params, xgb_train, self.num_rounds, watchlist, early_stopping_rounds=self.early_stop_rounds)

    def test(self, name):
        print("Xgboost testing...")

        train_data, train_labels, test_data = self.prepare_test_data(name)

        xgb_train = xgb.DMatrix(train_data, label=train_labels)
        xgb_test = xgb.DMatrix(test_data)

        num_rounds = 900
        watchlist = [(xgb_train, 'train')]
        model = xgb.train(self.params, xgb_train, num_rounds, watchlist)

        submit = model.predict(xgb_test)
        with open(config.output_prefix_path + 'xgboost_' +name +'-summit.txt', 'w') as fr:
            for sub in submit:
                fr.write(str(sub) + '\n')


if __name__ == "__main__":
    model = Xgboost()
    model.train('human_feature')
    #model.test('human_feature')



