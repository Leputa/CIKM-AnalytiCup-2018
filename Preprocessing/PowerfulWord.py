import sys
sys.path.append('../')

import os
import pickle
import numpy as np

from Preprocessing import Preprocess
from Config import config

class PowerfulWord():

    def __init__(self):
        self.preprocess = Preprocess.Preprocess()

    def generate_powerful_word(self):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        path = config.cache_prefix_path + 'powerful_words.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        words_power = {}
        _, _, left, right, labels =  self.preprocess.load_train_data('en')
        for i in range(len(left)):
            label = labels[i]
            q1_words = left[i]
            q2_words = right[i]
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(left)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        #words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
        with open(path, 'wb') as pkl:
            pickle.dump(words_power, pkl)
        return words_power


    def powerful_word_double_side(self):

        def init_powerful_word_oside(pword, thresh_num, thresh_rate):
            pword_oside = []
            pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
            pword_oside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
            return pword_oside

        sorted_words_power = self.generate_powerful_word()
        one_words_power = init_powerful_word_oside(sorted_words_power, 50, 0.8)
        return one_words_power

    def add_double_feature(self, tag):
        # 稀疏特征，可能不适合神经网络

        def extract_row(q1_words, q2_words, pword_dside):
            tags = []
            for word in pword_dside:
                if (word in q1_words) and (word in q2_words):
                    tags.append(1.0)
                else:
                    tags.append(0.0)
            return tags

        path = config.cache_prefix_path + tag + '_pword_dside.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        pword_dside = self.powerful_word_double_side()
        feature = []

        for i in range(len(left)):
            feature.append(extract_row(left[i], right[i], pword_dside))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def add_double_rate_feature(self, tag):
        def extract_row(q1_words, q2_words, pword_dict):

            num_least = 200
            rate = [1.0]
            share_words = list(set(q1_words).intersection(set(q2_words)))

            for word in share_words:
                if word not in pword_dict:
                    continue
                if pword_dict[word][0] * pword_dict[word][5] < num_least:
                    continue
                rate[0] *= (1.0 - pword_dict[word][6])
            rate = [1 - num for num in rate]
            return rate

        path = config.cache_prefix_path + tag + 'pword_dside_rate.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        pword_dside = self.generate_powerful_word()

        feature = []

        for i in range(len(left)):
            feature.append(extract_row(left[i], right[i], pword_dside))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def add_one_rate_feature(self, tag):
        def extract_row(q1_words, q2_words, pword_dict):

            num_least = 200
            rate = [1.0]
            share_words = list(set(q1_words).intersection(set(q2_words)))

            for word in share_words:
                if word not in pword_dict:
                    continue
                if pword_dict[word][0] * pword_dict[word][3] < num_least:
                    continue
                rate[0] *= (1.0 - pword_dict[word][4])
            rate = [1 - num for num in rate]
            return rate

        path = config.cache_prefix_path + tag + 'pword_oside_rate.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        pword_dside = self.generate_powerful_word()

        feature = []

        for i in range(len(left)):
            feature.append(extract_row(left[i], right[i], pword_dside))
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def powerful_word_one_side(self):
        # 不存在满足条件的这类词, 忽略这个特征
        def init_powerful_word_oside(pword, thresh_num, thresh_rate):
            pword_dside = []
            pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
            pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
            return pword_dside

        sorted_words_power = self.generate_powerful_word()
        double_words_power = init_powerful_word_oside(sorted_words_power, 30, 0.8)
        return double_words_power


    def load_left_right(self, tag):
        dic = {
            'train':'en',
            'dev': 'es'
        }
        if tag == 'train' or tag == 'dev':
            _, _, left, right, _ = self.preprocess.load_train_data(dic[tag])
        elif tag == 'test_a' or tag == 'test_b':
            left, right = self.preprocess.load_test(tag[-1].upper())

        return left, right

    def addtional_feature(self, tag, modeltype):
        dwords_rate = self.add_double_rate_feature(tag)
        owords_rate = self.add_one_rate_feature(tag)
        if modeltype == 'LexDecomp' or modeltype == 'Xgboost' or modeltype == 'LightGbm' or modeltype == 'FM_FTRL':
            return np.hstack([dwords_rate, owords_rate])



if __name__ == '__main__':
    model = PowerfulWord()
    drate = model.add_one_rate_feature('train')

