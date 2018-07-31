import sys
sys.path.append('../')

from Config import config
from Preprocessing import Preprocess
from Preprocessing import GraphFeature

import math



class Postprocess():
    def __init__(self):
        self.Proprocess = Preprocess.Preprocess()
        self.GraphFeture = GraphFeature.GraphFeature()

    def assert_one(self, tag):
        # 训练集18.8%不符合
        # 验证级完美
        print('asserting one labels')

        short_path_length = self.GraphFeture.get_shortest_path(tag, 'label')
        _, _, _, _, labels = self.Proprocess.get_index_data(tag)
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
        _, _, _, _, labels = self.Proprocess.get_index_data(tag)
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
        label_1_short_path_length = self.GraphFeture.get_shortest_path('test_b', 'label')
        label_2_short_path_length = self.GraphFeture.get_shortest_path('test_b', 'label_2')
        test_left, test_right = self.Proprocess.load_test('B')

        sub = []
        with open(filename, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                sub.append(float(line.strip()))

        cnt = 0
        sum = 0
        # 1 revise
        for i in range(len(sub)):
            if label_1_short_path_length[i] == 0:
                if label_2_short_path_length[i] == 1 and sub[i] < 0.99:
                    sum += (math.log(sub[i]))
                    sub[i] = 1
                if 'hablar' in test_left[i] or 'hablar' in test_right[i]:
                    continue
                if sub[i] > 0.8 and sub[i] < 0.99:
                    sum += (math.log(sub[i]) - math.log(0.99))
                    cnt += 1
                    sub[i] = 0.99
        print(cnt)
        print(sum)

        cnt = 0
        sum = 0
        # 0 revise
        for i in range(len(sub)):
            if label_2_short_path_length[i] == 0:
                assert  ' '.join(test_left[i]) == ' '.join(test_right[i])
                sum += (math.log(sub[i]))
                sub[i] = 1
                cnt += 1
        print(cnt)
        print(sum)

        with open(config.output_prefix_path + filename.split('.')[0] + 'revise.txt', 'w') as fr:
            for s in sub:
                fr.write(str(s) + '\n')

        print('ending...')

    def rescale(self, filename):
        def adj(x, te, tr):
            a = te / tr
            b = (1 - te) / (1 - tr)
            return a * x / (a * x + b * (1 - x))

        print("rescale......")
        sub = []
        with open(config.output_prefix_path + filename, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                sub.append(float(line.strip()))

        if filename ==  'ABCNN3-submit——0.40833.txt':
            sub_rescale = [adj(s, 0.12, 0.27) for s in sub]
        elif filename == 'xgboost_human_feature-summit——0.40224.txt':
            sub_rescale = [adj(s, 0.12, 0.235) for s in sub]
        elif filename == 'LexDecomp-submit——0.41079.txt':
            sub_rescale = [adj(s, 0.12, 0.247) for s in sub]

        with open(config.output_prefix_path + filename.split('-')[0] + '-rescale.txt', 'w') as fr:
            for s in sub_rescale:
                fr.write(str(s) + '\n')






if __name__ == '__main__':
    postprocess = Postprocess()
    # postprocess.assert_one('train')
    postprocess.graph_revise(config.output_prefix_path + 'blending.txt')
    # postprocess.assert_zero('train')
    #postprocess.rescale('xgboost_human_feature-summit——0.40224.txt')
