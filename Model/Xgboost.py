import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import Feature
from Preprocessing import PowerfulWord
from Preprocessing import GraphFeature
from Model.BaseMlModel import BaseMlModel

from scipy.sparse import csc_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
import xgboost as xgb

class Xgboost(BaseMlModel):

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.Feature = Feature.Feature()
        self.Powerfulwords = PowerfulWord.PowerfulWord()
        self.Graph = GraphFeature.GraphFeature()
        self.n_folds = 5

        self.params = {  'booster':'gbtree',
                         'max_depth':6,
                         'eta':0.03,
                         # 'max_bin':425,
                         # 'subsample_for_bin':50000,
                         'objective':'binary:logistic',
                         # 'min_split_gain':0,
                         #'min_child_weight':5,
                         'subsample':0.7,
                         'colsample_bytree':0.7,
                         'lambda':10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                         'alpha':1,   # L1正则化
                         'seed':2018,
                         'nthread':7,
                         'silent':True,
                         'gamma': 0.1,
                         #'scale_pos_weight': 0.25,  #测试的时候应该是0.12
                         'eval_metric':'logloss'
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

        num_rounds = 500
        watchlist = [(xgb_train, 'train')]
        model = xgb.train(self.params, xgb_train, num_rounds, watchlist)

        submit = model.predict(xgb_test)
        with open(config.output_prefix_path + str(num_rounds) + '_xgboost_' +name +'-summit.txt', 'w') as fr:
            for sub in submit:
                fr.write(str(sub) + '\n')

    def cv(self, name):
        print("交叉验证......")
        train_data, train_labels, test_data = self.prepare_test_data(name)
        folds = KFold(n_splits=self.n_folds, random_state=2018, shuffle=True)

        auc = 0
        iter = 0

        oof = np.zeros(train_data.shape[0])
        sub = np.zeros(test_data.shape[0])
        dtest = xgb.DMatrix(test_data)

        train_data = csc_matrix(train_data)    # coo_matrix不能分片
        #train_labels = np.array(train_labels) # 直接改成array来分片有bug，想不通为什么

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_data, train_labels)):

            t_labels, v_labels = [], []
            for idx in trn_idx:
                t_labels.append(train_labels[idx])
            for idx in val_idx:
                v_labels.append(train_labels[idx])


            dtrain = xgb.DMatrix(train_data[trn_idx], label=t_labels)
            dvalid = xgb.DMatrix(train_data[val_idx], label=v_labels)
            watchlist = [(dtrain, 'train'), (dvalid, 'val')]
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round = self.num_rounds,
                evals = watchlist,
                early_stopping_rounds = self.early_stop_rounds,
                verbose_eval = 100,
            )
            oof[val_idx] = model.predict(dvalid, ntree_limit=model.best_iteration)
            tmp_auc = roc_auc_score(v_labels, oof[val_idx])
            auc += tmp_auc

            print("\t Fold %d : %.6f in %3d rounds" % (n_fold + 1, tmp_auc, model.best_iteration))
            sub += model.predict(dtest, ntree_limit=model.best_iteration)

        sub /= self.n_folds
        iter /= self.n_folds
        print('平均迭代次数： ' + str(iter))

        with open(config.output_prefix_path + 'xgboost_oof.txt') as fr:
            for i in range(len(oof)):
                fr.write(str(oof[i]) + '/n')

        with open(config.output_prefix_path + 'xgboost_cv_sub.txt') as fr:
            for i in range(len(sub)):
                fr.write(str(sub[i]) + '/n')







if __name__ == "__main__":
    model = Xgboost()
    # model.train('human_feature')
    model.test('human_feature')
    # model.cv('human_feature')



