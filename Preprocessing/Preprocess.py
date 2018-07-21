import sys
sys.path.append('../')

import os
import pickle
import copy
import numpy as np
from tqdm import tqdm


from Config import config
from Config import tool
from Preprocessing.Tokenizer import *
from Preprocessing.WordDict import *


class Preprocess():
    def __init__(self):
        self.max_es_length = 50
        self.max_en_length = 50
        self.tokenizer = Tokenizer()

    def load_train_data(self, tag='en'):
        print("导入训练数据")
        if tag == 'en':
            path = config.TOKEN_TRAIN
        elif tag == 'es':
            path = config.TOKEN_VAL

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag == 'en':
            data_path = config.EN_TRAIN_FILE
        elif tag == 'es':
            data_path = config.ES_TRAIN_FILE

        en_sentence_left = []
        en_sentence_right = []
        es_sentence_left = []
        es_sentence_right = []
        labels = []

        with open(data_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                lineList = line.split('\t')
                assert len(lineList) == 5
                # 句子切分
                if tag == 'en':
                    # 英文
                    tmp_en_left = self.tokenizer.en_str_clean(lineList[0])
                    tmp_en_right = self.tokenizer.en_str_clean(lineList[2])

                    # 西班牙文
                    tmp_es_left = self.tokenizer.es_str_clean(lineList[1])
                    tmp_es_right = self.tokenizer.es_str_clean(lineList[3])

                elif tag == 'es':
                    tmp_en_left = self.tokenizer.en_str_clean(lineList[1])
                    tmp_en_right = self.tokenizer.en_str_clean(lineList[3])

                    tmp_es_left = self.tokenizer.es_str_clean(lineList[0])
                    tmp_es_right = self.tokenizer.es_str_clean(lineList[2])

                # 先不要加trick
                # 添加公共序列
                # en_common_list = tool.LCS(tmp_en_left, tmp_en_right)
                # tmp_en_left.extend(en_common_list)
                # tmp_en_right.extend(en_common_list)
                #
                # es_common_list = tool.LCS(tmp_es_left, tmp_es_right)
                # tmp_es_left.extend(es_common_list)
                # tmp_es_right.extend(es_common_list)

                en_sentence_left.append(tmp_en_left)
                en_sentence_right.append(tmp_en_right)
                es_sentence_left.append(tmp_es_left)
                es_sentence_right.append(tmp_es_right)

                labels.append(int(lineList[4]))

        with open(path, 'wb') as pkl:
            pickle.dump((en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, labels), pkl)

        return (en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, labels)


    def load_test(self, lang = 'es'):
        print("导入测试数据")

        if lang == 'es':
            path = config.TOKEN_TEST
        elif lang == 'en':
            path = config.cache_prefix_path + 'token_test_en.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        sentence_left = []
        sentence_right = []

        if lang == 'es':
            file_path = config.TEST_FiLE
        elif lang == 'en':
            file_path = config.data_prefix_path + 'test_en.txt'

        with open(file_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()

            for line in lines:
                lineList = line.split('\t')
                if lang == 'es':
                    tmp_left = self.tokenizer.es_str_clean(lineList[0])
                    tmp_right = self.tokenizer.es_str_clean(lineList[1])
                elif lang == 'en':
                    tmp_left = self.tokenizer.en_str_clean(lineList[0])
                    tmp_right = self.tokenizer.en_str_clean(lineList[1])

                # common_list = tool.LCS(tmp_left, tmp_right)
                # tmp_left.extend(common_list)
                # tmp_right.extend(common_list)

                sentence_left.append(tmp_left)
                sentence_right.append(tmp_right)

        with open(path, 'wb') as pkl:
            pickle.dump((sentence_left, sentence_right), pkl)

        return (sentence_left, sentence_right)

    def es2index(self, lang='es'):
        print("建立词到索引的字典")
        word2index = WordDict()

        if lang == 'es':
            path = config.cache_prefix_path + 'Es2IndexDic.pkl'
        elif lang == 'en':
            path = config.cache_prefix_path + 'En2IndexDic.pkl'
        if os.path.exists(path):
            return word2index.loadWord2IndexDic(lang)

        if lang == 'es':
            _, _, train_left, train_right, _  = self.load_train_data('en')
            _, _, dev_left, dev_right, _ = self.load_train_data('es')
        elif lang == 'en':
            train_left, train_right, _, _, _ = self.load_train_data('en')
            dev_left, dev_right, _, _, _ = self.load_train_data('es')

        test_left, test_right = self.load_test(lang)

        for i in range(len(train_left)):
            for word in train_left[i]:
                word2index.add_word(word, lang)
            for word in train_right[i]:
                word2index.add_word(word, lang)

        for i in range(len(dev_left)):
            for word in dev_left[i]:
                word2index.add_word(word, lang)
            for word in dev_right[i]:
                word2index.add_word(word, lang)

        for i in range(len(test_left)):
            for word in test_left[i]:
                word2index.add_word(word, lang)
            for word in test_right[i]:
                word2index.add_word(word, lang)

        word2index.saveWord2IndexDic(lang)
        return word2index.Es2IndexDic


    def get_index_data(self, tag):
        print("将语料转化为索引表示")

        path = config.cache_prefix_path + tag + '_index.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        dic = {
            'train':'en',
            'dev': 'es'
        }

        if tag == 'train' or tag == 'dev':
            left_en, right_en, left_es, right_es, labels = self.load_train_data(dic[tag])
        if tag == 'test':
            left_es, right_es = self.load_test('es')
            left_en, right_en = self.load_test('en')

        es_word2index = self.es2index('es')
        en_word2index = self.es2index('en')

        es_left_index, es_right_index = [], []
        en_left_index, en_right_index = [], []

        for i in range(len(left_es)):
            es_left_index.append([es_word2index.get(word, 1) for word in left_es[i]])
            es_right_index.append([es_word2index.get(word, 1) for word in right_es[i]])

        for i in range(len(left_en)):
            en_left_index.append([en_word2index.get(word, 1) for word in left_en[i]])
            en_right_index.append([en_word2index.get(word, 1) for word in right_en[i]])

        if tag == 'train' or tag == 'dev':
            with open(path, 'wb') as pkl:
                pickle.dump((en_left_index, en_right_index, es_left_index, es_right_index, labels), pkl)
            return  en_left_index, en_right_index, es_left_index, es_right_index, labels

        elif tag == 'test':
            with open(path, 'wb') as pkl:
                pickle.dump((en_left_index, en_right_index, es_left_index, es_right_index), pkl)
            return en_left_index, en_right_index, es_left_index, es_right_index


    def get_index_padding(self, tag):
        print("padding")
        path = config.cache_prefix_path + tag + '_index_padding.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag == 'train' or tag == 'dev':
            en_left_index, en_right_index, es_left_index, es_right_index, labels = self.get_index_data(tag)
        if tag == 'test':
            en_left_index, en_right_index, es_left_index, es_right_index = self.get_index_data(tag)

        en_left_index_padding = copy.deepcopy(en_left_index)
        en_right_index_padding = copy.deepcopy(en_right_index)
        es_left_index_padding = copy.deepcopy(es_left_index)
        es_right_index_padding = copy.deepcopy(es_right_index)

        for i in range(len(en_left_index)):
            if len(en_left_index[i]) < self.max_en_length:
                en_left_index_padding[i] += [0] * (self.max_en_length - len(en_left_index_padding[i]))
            else:
                en_left_index_padding[i] = en_left_index_padding[i][:self.max_en_length]

            if len(en_right_index[i]) < self.max_en_length:
                en_right_index_padding[i] += [0] * (self.max_en_length - len(en_right_index_padding[i]))
            else:
                en_right_index_padding[i] = en_right_index_padding[i][:self.max_en_length]

        for i in range(len(es_left_index)):
            if len(es_left_index[i]) < self.max_es_length:
                es_left_index_padding[i] += [0] * (self.max_es_length - len(es_left_index_padding[i]))
            else:
                es_left_index_padding[i] = es_left_index_padding[i][:self.max_es_length]

            if len(es_right_index[i]) < self.max_es_length:
                es_right_index_padding[i] += [0] * (self.max_es_length - len(es_right_index_padding[i]))
            else:
                es_right_index_padding[i] = es_right_index_padding[i][:self.max_es_length]

        if tag == 'train' or tag == 'dev':
            with open(path, 'wb') as pkl:
                pickle.dump((en_left_index_padding, en_right_index_padding, es_left_index_padding, es_right_index_padding, labels), pkl)
            return (en_left_index_padding, en_right_index_padding, es_left_index_padding, es_right_index_padding, labels)
        if tag == 'test':
            with open(path, 'wb') as pkl:
                pickle.dump((en_left_index_padding, en_right_index_padding, es_left_index_padding, es_right_index_padding), pkl)
            return (en_left_index_padding, en_right_index_padding, es_left_index_padding, es_right_index_padding)


    def get_length(self, tag):
        print("get length")
        path = config.cache_prefix_path + tag + '_length.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag == 'train':
            en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, _ = self.load_train_data('en')
        elif tag == 'dev':
            en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, _ = self.load_train_data('es')
        elif tag == 'test':
            es_sentence_left, es_sentence_right = self.load_test()
            en_sentence_left, en_sentence_right = self.load_test('en')

        en_left_length = [min(len(sentence), self.max_en_length) for sentence in en_sentence_left]
        en_right_length = [min(len(sentence), self.max_en_length) for sentence in en_sentence_right]

        es_left_length = [min(len(sentence), self.max_es_length) for sentence in es_sentence_left]
        es_right_length = [min(len(sentence), self.max_es_length) for sentence in es_sentence_right]

        with open(path, 'wb') as pkl:
            pickle.dump((en_left_length, en_right_length, es_left_length, es_right_length), pkl)
        return (en_left_length, en_right_length, es_left_length, es_right_length)


    def load_translation_data(self):
        print("loading translation data")

        path = config.cache_prefix_path + 'translation_token.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        data_path = config.TRANSLATE_FILE
        es = []
        en = []
        with open(data_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                line = line.split("\t")
                es.append(self.tokenizer.es_str_clean(line[0]))
                en.append(self.tokenizer.en_str_clean(line[1]))

        with open(path, 'wb') as pkl:
            pickle.dump((es, en), pkl)
        return (es, en)

    def load_all_data(self):
        print('loading all spanish&english')

        path = config.cache_prefix_path + 'all_token.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        en = []
        es = []

        en_left_train, en_right_train, es_left_train, es_right_train, _ = self.load_train_data('en')
        en_left_dev, en_right_dev, es_left_dev, es_right_dev, _ = self.load_train_data('es')
        es_left_test, es_right_test = self.load_test()
        en_left_test, en_right_test = self.load_test('en')

        es_trans, en_trans = self.load_translation_data()

        en.extend(en_left_train)
        en.extend(en_right_train)
        en.extend(en_left_dev)
        en.extend(en_right_dev)
        en.extend(en_left_test)
        en.extend(en_right_test)
        en.extend(en_trans)

        es.extend(es_left_train)
        es.extend(es_right_train)
        es.extend(es_left_dev)
        es.extend(es_right_dev)
        es.extend(es_left_test)
        es.extend(es_right_test)
        es.extend(es_trans)

        with open(path, 'wb') as pkl:
            pickle.dump((es, en), pkl)
        return es, en

    def swap_data(self, tag1, tag2):
        print('Swapping data')

        path = config.cache_prefix_path + tag1 + '_' + tag2 + '_swap.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        dic = {
            'train':'en',
            'dev': 'es'
        }

        if tag2 == 'token':
            en_left, en_right, es_left, es_right, labels = self.load_train_data(dic[tag1])
        elif tag2 == 'index':
            en_left, en_right, es_left, es_right, labels = self.get_index_data(tag1)
        elif tag2 == 'padding':
            en_left, en_right, es_left, es_right, labels = self.get_index_padding(tag1)

        with open(path, 'wb') as pkl:
            pickle.dump((en_right, en_left, es_right, es_left, labels), pkl)

        return (en_right, en_left, es_right, es_left, labels)


if __name__ == '__main__':
    p = Preprocess()

    p.load_test('en')
    p.es2index('en')
    p.get_index_data('train')
    p.get_index_data('dev')
    p.get_index_data('test')
    p.get_index_padding('train')
    p.get_index_padding('dev')
    p.get_index_padding('test')
    p.swap_data('train', 'padding')
    p.swap_data('dev', 'padding')
    p.load_all_data()
    p.get_length('train')
    p.get_length('dev')
    p.get_length('test')
