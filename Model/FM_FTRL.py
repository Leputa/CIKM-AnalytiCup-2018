import sys
sys.path.append('../')

from Config import config
from Model.BaseMlModel import BaseMlModel

from sklearn import preprocessing
from sklearn.metrics import log_loss
from wordbatch.models import FM_FTRL
from sklearn.metrics import roc_curve, auc, roc_auc_score

class FM_FTRL_Model(BaseMlModel):

    def __init__(self):
        super().__init__()


    def train(self, tag):
        print("FM_FTRL training")

        train_data, train_labels, dev_data, dev_labels = self.prepare_train_data(tag, 'FM_FTRL')
        train_data = preprocessing.scale(train_data, with_mean=False)
        dev_data = preprocessing.scale(dev_data, with_mean=False)

        self.clf = FM_FTRL(
            alpha=0.001,  # w0和w的FTRL超参数alpha
            beta=0.01,  # w0和w的FTRL超参数beta
            L1=0.00001,  # w0和w的L1正则
            L2=0.1,  # w0和w的L2正则
            D=train_data.shape[1],

            alpha_fm=0.001,  # v的FTRL超参数alpha
            L2_fm=0.0,       # v的L2正则

            init_fm=0.01,
            D_fm=200,
            e_noise=0.0001,
            # iters=5,
            inv_link="identity",
            threads=7,
        )
        self.clf.fit(train_data, train_labels)

        y_train = self.clf.predict(train_data)
        y_val = self.clf.predict(dev_data)

        print('train_logloss: ' + str(log_loss(train_labels, y_train)))
        print("val_logloss: " + str(log_loss(dev_labels, y_val)))

        print("train_auc: " + str(roc_auc_score(train_labels, y_train)))
        print("val_auc: " + str(roc_auc_score(dev_labels, y_val)))

    def test(self, name):
        print("FM_FTRL testing...")

        train_data, train_labels, test_data = self.prepare_test_data(name, 'FM_FTRL')

        self.clf = FM_FTRL(
            alpha=0.01,  # w0和w的FTRL超参数alpha
            beta=0.01,   # w0和w的FTRL超参数beta
            L1=0,        # w0和w的L1正则
            L2=0,        # w0和w的L2正则
            D=train_data.shape[1],

            alpha_fm=0.005,  # v的FTRL超参数alpha
            L2_fm=0.01,      # v的L2正则

            init_fm=0.01,
            D_fm=2,
            e_noise=0.0001,
            iters=3,
            inv_link="sigmoid",
            threads=7,
        )

        self.clf.fit(train_data, train_labels)

        submit = self.clf.predict(test_data)
        with open(config.output_prefix_path + 'FM_FTRL_' + name +'-summit.txt', 'w') as fr:
            for sub in submit:
                fr.write(str(sub) + '\n')


if __name__ == '__main__':
    model = FM_FTRL_Model()
    model.train('human_feature')