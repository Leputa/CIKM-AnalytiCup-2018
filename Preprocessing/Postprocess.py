import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import GraphFeature



class Postprocess():
    def __init__(self):
        self.Proprocess = Preprocess.Preprocess()
        self.GraphFeture = GraphFeature.GraphFeature()

    def assert_one(self, tag):
        # 训练集18.8%不符合
        # 验证级完美
        print('asserting one labels')

        short_path_length = self.GraphFeture.get_shortest_path(tag, 'label')
        _, _, labels = self.Proprocess.get_es_index_data(tag)
        sum = 0
        cnt = 0

        assert len(short_path_length) == len(labels)
        for i in range(len(short_path_length)):
            if short_path_length[i] == 0:
                sum += 1
                if labels[i] != 1:
                    cnt += 1
                    print(i+1)
        print(sum)
        print(cnt/sum)

    def assert_zero(self, tag):
        # 训练集上有11.7%不符合
        # 验证级完美
        print('asserting zero labels')
        short_path_length = self.GraphFeture.get_shortest_path(tag, 'label_2')
        _, _, labels = self.Proprocess.get_es_index_data(tag)
        sum = 0
        cnt = 0
        assert len(short_path_length) == len(labels)
        for i in range(len(short_path_length)):
            if short_path_length[i] > 100 and short_path_length[i]%100 != 0:
                sum += 1
                if labels[i] != 0:
                    cnt += 1
                    print(i+1)
        print(cnt/sum)

    def graph_revise(self, filename):
        label_1_short_path_length = self.GraphFeture.get_shortest_path('test', 'label')
        label_2_short_path_length = self.GraphFeture.get_shortest_path('test', 'label_2')

        sub = []
        with open(filename, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                sub.append(float(line.strip()))

        # 1 revise
        for i in range(len(sub)):
            if label_1_short_path_length[i] == 0:
                if sub[i] < 0.99:
                    sub[i] = 0.99
        # 0 revise
        for i in range(len(sub)):
            if label_2_short_path_length[i] > 100 and label_2_short_path_length[i]%100 != 0:
                if sub[i] > 0.01:
                    sub[i] = 0.01

        with open(config.output_prefix_path + 'xgboost_revise.txt', 'w') as fr:
            for s in sub:
                fr.write(str(s) + '\n')

        print('ending...')




if __name__ == '__main__':
    postprocess = Postprocess()
    postprocess.graph_revise(config.output_prefix_path + 'xgboost_human_feature-summit——0.40857.txt')
    #postprocess.assert_zero('dev')
