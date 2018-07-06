import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import Feature
from Preprocessing import PowerfulWord
from Model.BaseMlModel import BaseMlModel

import lightgbm as lgb

class LightGbm(BaseMlModel):

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.Feature = Feature.Feature()
        self.Powerfulwords = PowerfulWord.PowerfulWord()

        self.params = {
                # 核心参数
                'task': 'train',  # 设置是否是训练任务
                'objective': 'binary',  # 设置目标  =='application': 'binary',
                'boosting_type': 'gbdt',  # 设置模型
                'num_iterations': 5000,  # 最大循环次数
                'learning_rate': 0.05,  # 学习率
                'num_leaves': 35,  # 设置一棵树最多有几个叶子，越大越容易过拟合
                # 'tree_learner':'tree_learner' # 设置并行学习
                'num_threads': 7,  # 指定多少个线程
                # 'device':'cpu',# 指定设备GPU／cpu

                # 学习控制参数
                'max_depth': 6,  # 限制树模型的最大深度，用于减少过拟合，<0表示没有限制
                'min_data_in_leaf': 10,# 一个叶子中最少数据个数，用于制止过拟合
                # 'min_sum_hessian_in_leaf':0.00001,# minimal sum hessian in one leaf,制止过拟合
                # 'feature_fraction':1.0,# 随机抽取的特征列的占比，用于加快训练和防止过拟合
                'feature_fraction_seed':2018,# 随机抽取的特征列的随机产生种子
                # 'bagging_fraction':1.0,# 随机选择部分数据的占比，用于加快训练和防止过拟合
                # 'bagging_freq':1,# 每隔k次就进行随机bagging，k=0表示不bagging
                'bagging_seed': 2018,# bagging的随机产生种子
                'lambda_l1':1,# L1 regularization
                'lambda_l2':10,# L2 regularization
                # 'min_split_gain':0,# 分裂一次最少要获取多少收益
                # 'max_cat_threshold':32,# limit the max threshold points in categorical features
                # 'cat_smooth':10,# 减少类别变量的噪声污染，对于少数据的类别
                #'cat_l2': 4,  # 类别分类的L2正则
                # 'max_cat_to_onehot':4,# 当一个类别总数少于等于4，使用one-vs-other分裂算法
                'colsample_bytree': 0.7,  # 在建立树时对特征采样的比例。缺省值为1
                'subsample': 0.8,  # 用于训练模型的子样本占整个样本集合的比例。
                # IO参数
                # 'max_bin':255, #max number of bins that feature values will be bucketed in,小的值可能减少精度，但增加泛化能力
                # 'min_data_in_bin':3, # min number of data inside one bin, use this to avoid one-data-one-bin (may over-fitting)
                # 'data_random_seed':1, # random seed for data partition in parallel learning (not include feature parallel)
                # 'input_model':"" # 训练任务将会继续训练，预测任务直接预测结果
                # 'is_sparse':True # 离散优化
                'verbose': 0,  # 是否报出信息，<0 = Fatal, =0 = Error (Warn), >0 = Info

                # 'bin_construct_sample_cnt':200000 # will give better training result when set this larger, but will increase data loading time
                # 'num_iteration_predict':-1, # 只用于预测任务，指定用多少训练迭代次数来进行预测，-1表示无限制
                # 'pred_early_stop':false,# if true will use early-stopping to speed up the prediction. May affect the accuracy
                # 'use_missing':true,# set to false to disable the special handle of missing value

                # 目标参数
                # 'is_unbalance':True,# set this to true if training data are unbalance

                # 指标参数
                'eval_metric': 'logloss',  # logloss
        }
        self.early_stop_rounds = 100


    def train(self, tag):
        print("LightGbm training")

        train_data, train_labels, dev_data, dev_labels = self.prepare_train_data(tag)

        lgb_train = lgb.Dataset(data = train_data, label=train_labels)
        lgb_val = lgb.Dataset(data = dev_data, label = dev_labels)

        gbm = lgb.train(self.params, lgb_train,
                        valid_sets=[lgb_train, lgb_val], early_stopping_rounds=self.early_stop_rounds,
                        verbose_eval=1)

    def test(self, name):
        print("LightGbm testing...")

        train_data, train_labels, test_data = self.prepare_test_data(name)

        lgb_train = lgb.Dataset(data=train_data, label=train_labels)
        lgb_test = lgb.Dataset(test_data)

        model = lgb.train(self.params, lgb_train, valid_sets=[lgb_train],
                          num_boost_round=1900, verbose_eval=1)

        submit = model.predict(test_data)
        with open(config.output_prefix_path + 'lightgbm_' + name +'-summit.txt', 'w') as fr:
            for sub in submit:
                fr.write(str(sub) + '\n')


if __name__ == "__main__":
    model = LightGbm()
    model.train('human_feature')
    #model.test('human_feature')