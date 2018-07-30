import sys
sys.path.append('../')

import os
import pickle
import copy
import numpy as np
from tqdm import tqdm
import io
import string
import Levenshtein

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


    def load_test(self, tag):
        print("导入测试数据")

        if tag == 'B':
            path = config.TOKEN_TEST_B
        elif tag == 'A':
            path = config.TOKEN_TEST_A
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        sentence_left = []
        sentence_right = []

        if tag == 'B':
            file_path = config.TEST_FILE_B
        elif tag == 'A':
            file_path = config.TEST_FILE_A

        with open(file_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in tqdm(lines):
                lineList = line.split('\t')
                tmp_left = self.tokenizer.es_str_clean(lineList[0])
                tmp_right = self.tokenizer.es_str_clean(lineList[1])

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
            _, _, train_left, train_right, _  = self.replace_words('train')
            _, _, dev_left, dev_right, _ = self.replace_words('dev')
            test_left_b, test_right_b = self.replace_words('test_b')
            test_left_a, test_right_a = self.replace_words('test_a')
        elif lang == 'en':
            train_left, train_right, _, _, _ = self.load_train_data('en')
            dev_left, dev_right, _, _, _ = self.load_train_data('es')

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

        if lang == 'es':
            for i in range(len(test_left_a)):
                for word in test_left_a[i]:
                    word2index.add_word(word, lang)
                for word in test_right_a[i]:
                    word2index.add_word(word, lang)
            for i in range(len(test_left_b)):
                for word in test_left_b[i]:
                    word2index.add_word(word, lang)
                for word in test_right_b[i]:
                    word2index.add_word(word, lang)

        word2index.saveWord2IndexDic(lang)
        return word2index.Es2IndexDic


    def get_index_data(self, tag):
        print("将语料转化为索引表示")

        path = config.cache_prefix_path + tag + '_index.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag == 'train' or tag == 'dev':
            left_en, right_en, left_es, right_es, labels = self.replace_words(tag)
        if tag == 'test_a' or tag == 'test_b':
            left_es, right_es = self.replace_words(tag)

        es_word2index = self.es2index('es')
        en_word2index = self.es2index('en')

        es_left_index, es_right_index = [], []
        en_left_index, en_right_index = [], []

        for i in range(len(left_es)):
            es_left_index.append([es_word2index.get(word, 1) for word in left_es[i]])
            es_right_index.append([es_word2index.get(word, 1) for word in right_es[i]])

        if tag == 'train' or tag == 'dev':
            for i in range(len(left_en)):
                en_left_index.append([en_word2index.get(word, 1) for word in left_en[i]])
                en_right_index.append([en_word2index.get(word, 1) for word in right_en[i]])

            with open(path, 'wb') as pkl:
                pickle.dump((en_left_index, en_right_index, es_left_index, es_right_index, labels), pkl)
            return  en_left_index, en_right_index, es_left_index, es_right_index, labels

        elif tag == 'test_a' or tag == 'test_b':
            with open(path, 'wb') as pkl:
                pickle.dump((es_left_index, es_right_index), pkl)
            return es_left_index, es_right_index


    def get_index_padding(self, tag):
        print("padding")
        path = config.cache_prefix_path + tag + '_index_padding.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag == 'train' or tag == 'dev':
            en_left_index, en_right_index, es_left_index, es_right_index, labels = self.get_index_data(tag)
        if tag == 'test_a' or tag == 'test_b':
            es_left_index, es_right_index = self.get_index_data(tag)

        if tag == 'train' or tag == 'dev':
            en_left_index_padding = copy.deepcopy(en_left_index)
            en_right_index_padding = copy.deepcopy(en_right_index)

            for i in range(len(en_left_index)):
                if len(en_left_index[i]) < self.max_en_length:
                    en_left_index_padding[i] += [0] * (self.max_en_length - len(en_left_index_padding[i]))
                else:
                    en_left_index_padding[i] = en_left_index_padding[i][:self.max_en_length]

                if len(en_right_index[i]) < self.max_en_length:
                    en_right_index_padding[i] += [0] * (self.max_en_length - len(en_right_index_padding[i]))
                else:
                    en_right_index_padding[i] = en_right_index_padding[i][:self.max_en_length]

        es_left_index_padding = copy.deepcopy(es_left_index)
        es_right_index_padding = copy.deepcopy(es_right_index)

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
        if tag == 'test_a' or tag == 'test_b':
            with open(path, 'wb') as pkl:
                pickle.dump((es_left_index_padding, es_right_index_padding), pkl)
            return (es_left_index_padding, es_right_index_padding)


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
        elif tag == 'test_a' or tag == 'test_b':
            es_sentence_left, es_sentence_right = self.load_test(tag[-1].upper())

        es_left_length = [min(len(sentence), self.max_es_length) for sentence in es_sentence_left]
        es_right_length = [min(len(sentence), self.max_es_length) for sentence in es_sentence_right]

        if tag == 'train' or tag == 'dev':
            en_left_length = [min(len(sentence), self.max_en_length) for sentence in en_sentence_left]
            en_right_length = [min(len(sentence), self.max_en_length) for sentence in en_sentence_right]

            with open(path, 'wb') as pkl:
                pickle.dump((en_left_length, en_right_length, es_left_length, es_right_length), pkl)
            return (en_left_length, en_right_length, es_left_length, es_right_length)

        elif tag == 'test_a' or tag == 'test_b':
            with open(path, 'wb') as pkl:
                pickle.dump((es_left_length, es_right_length), pkl)
            return (es_left_length, es_right_length)


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

    def load_replace_translation_data(self):
        print("loading replace translation data")

        path = config.cache_prefix_path + 'replace_translation_token.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        es, en = self.load_translation_data()
        all_unigram_cnt = self.generate_unigram()
        all_bigram_cnt = self.generate_bigram()
        lack_words_set = self.get_lack_words_set()
        word_candidate_dic = self.get_candidate_words()

        for i in tqdm(range(len(es))):
            s = es[i]
            s = ['/s'] + s + ['/s']
            for j in range(1, len(s)-1):
                if s[j] in lack_words_set:
                    cand_words = word_candidate_dic.get(s[j])
                    if cand_words == None:
                        continue
                    elif len(cand_words) == 1:
                        s[j] = cand_words[0]
                    else:
                        maxlike_word = None
                        max_proba = 0
                        for cw in cand_words:
                            bigram_word1 = s[j] + '_' + cw
                            bigram_word2 = cw + '_' + s[j]
                            proba = (all_bigram_cnt.get(bigram_word1, 0) + 1) * (all_bigram_cnt.get(bigram_word2, 0) + 1)
                            if proba > max_proba:
                                max_proba = proba
                                maxlike_word = cw
                            if proba == max_proba:
                                if all_unigram_cnt[cw] > all_unigram_cnt[maxlike_word]:
                                    maxlike_word = cw
                        s[j] = maxlike_word
            es[i] = s[1:-1]

        with open(path, 'wb') as pkl:
            pickle.dump((es, en), pkl)

        return es

    def load_all_replace_data(self):
        print('loading all replace spanish&english')

        path = config.cache_prefix_path + 'replace_all_token.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        en = []
        es = []

        en_left_train, en_right_train, es_left_train, es_right_train, _ = self.replace_words('train')
        en_left_dev, en_right_dev, es_left_dev, es_right_dev, _ = self.replace_words('dev')
        es_left_test_a, es_right_test_a = self.replace_words('test_a')
        es_left_test_b, es_right_test_b = self.replace_words('test_b')

        es_trans, en_trans = self.load_replace_translation_data()

        en.extend(en_left_train)
        en.extend(en_right_train)
        en.extend(en_left_dev)
        en.extend(en_right_dev)
        en.extend(en_trans)

        es.extend(es_left_train)
        es.extend(es_right_train)
        es.extend(es_left_dev)
        es.extend(es_right_dev)
        es.extend(es_left_test_a)
        es.extend(es_right_test_a)
        es.extend(es_left_test_b)
        es.extend(es_right_test_b)
        es.extend(es_trans)

        with open(path, 'wb') as pkl:
            pickle.dump((es, en), pkl)
        return es, en

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
        es_left_test_a, es_right_test_a = self.load_test('A')
        es_left_test_b, es_right_test_b = self.load_test('B')

        es_trans, en_trans = self.load_translation_data()

        en.extend(en_left_train)
        en.extend(en_right_train)
        en.extend(en_left_dev)
        en.extend(en_right_dev)
        en.extend(en_trans)

        es.extend(es_left_train)
        es.extend(es_right_train)
        es.extend(es_left_dev)
        es.extend(es_right_dev)
        es.extend(es_left_test_a)
        es.extend(es_right_test_a)
        es.extend(es_left_test_b)
        es.extend(es_right_test_b)
        es.extend(es_trans)

        with open(path, 'wb') as pkl:
            pickle.dump((es, en), pkl)
        return es, en

    def swap_data(self, tag1, tag2):
        # 不涉及test_数据
        print('Swapping data')

        path = config.cache_prefix_path + tag1 + '_' + tag2 + '_swap.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag2 == 'token':
            en_left, en_right, es_left, es_right, labels = self.replace_words(tag1)
        elif tag2 == 'index':
            en_left, en_right, es_left, es_right, labels = self.get_index_data(tag1)
        elif tag2 == 'padding':
            en_left, en_right, es_left, es_right, labels = self.get_index_padding(tag1)

        with open(path, 'wb') as pkl:
            pickle.dump((en_right, en_left, es_right, es_left, labels), pkl)

        return (en_right, en_left, es_right, es_left, labels)

    def get_words(self):
        path = config.cache_prefix_path + 'words_set.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        es = self.load_all_data()[0]

        words_set = set()
        for s in tqdm(es):
            words_set |= set(s)

        with open(path, 'wb') as pkl:
            pickle.dump(words_set, pkl)

        return words_set

    def get_vec_words_set(self):
        path = config.cache_prefix_path + 'vec_words_set.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        vec_words = set()
        i = 0
        with io.open(config.ES_EMBEDDING_MATRIX, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for line in tqdm(f):
                ls = line.split()
                if ls[1][-1].isdigit():
                    vec_words.add(ls[0])

        with open(path, 'wb') as pkl:
            pickle.dump(vec_words, pkl)

        return vec_words

    def get_lack_words_set(self):
        path = config.cache_prefix_path + 'lack_words_set.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        words_set = self.get_words()
        vec_words_set = self.get_vec_words_set()

        lack_set = set()
        for word in words_set:
            # ID不做处理
            if any(ch in string.digits for ch in word)==True:
                continue
            # 规则处理
            if word.endswith('adar'):
                if word[:-1] in vec_words_set:
                    continue
                elif word[:-1]+'do' in vec_words_set:
                    continue
                elif word[:-1]+'ds' in vec_words_set:
                    continue
                elif word[:-1]+'os' in vec_words_set:
                    continue
                elif word[:-1]+'o' in vec_words_set:
                    continue
                elif word[:-1]+'s' in vec_words_set:
                    continue

            if word not in vec_words_set:
                lack_set.add(word)

        with open(path, 'wb') as pkl:
            pickle.dump(lack_set, pkl)

        return lack_set

    def get_candidate_words(self):
        path = config.cache_prefix_path + 'candidate_words.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        lack_words_set = self.get_lack_words_set()
        words_set = self.get_words()

        word_candidate_dic = {}
        word_in_doc_not_lack = words_set - lack_words_set

        for word in tqdm(lack_words_set):
            min_dist = float('inf')
            candidates = []
            closet_words = []
            for w in word_in_doc_not_lack:
                d = Levenshtein.distance(word, w)
                if d < min_dist:
                    min_dist = d
                    closet_words = [w]
                if d == min_dist:
                    closet_words.append(w)
                if d <= 2:
                    candidates.append(w)
            if min_dist/len(word) > 0.3 and min_dist > 2:
                continue
            candidates = list(set(candidates)|set(closet_words))
            word_candidate_dic[word] = candidates

        with open(path, 'wb') as pkl:
            pickle.dump(word_candidate_dic, pkl)

        return word_candidate_dic

    def generate_unigram(self):
        data = self.load_all_data()[0]

        s_set = set()
        idf = {}
        for s in tqdm(data):
            if ' '.join(s) not in s_set:
                s_set.add(' '.join(s))
                for word in s:
                    idf[word] = idf.get(word, 0) + 1
        return idf

    def generate_bigram(self):
        data = self.load_all_data()[0]

        s_set = set()
        idf = {}
        for s in tqdm(data):
            if ' '.join(s) not in s_set:
                s_set.add(' '.join(s))
                for i in range(len(s) - 1):
                    w = s[i] + '_' + s[i + 1]
                    idf[w] = idf.get(w, 0) + 1
        return idf

    def replace_words(self, tag):
        print('replacing words....')
        path = config.cache_prefix_path + tag + '_replace_token.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        if tag == 'train':
            en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, _ = self.load_train_data('en')
        elif tag == 'dev':
            en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, _ = self.load_train_data('es')
        elif tag == 'test_b':
            es_sentence_left, es_sentence_right = self.load_test('B')
        elif tag == 'test_a':
            es_sentence_left, es_sentence_right = self.load_test('A')

        all_unigram_cnt = self.generate_unigram()
        all_bigram_cnt = self.generate_bigram()
        lack_words_set = self.get_lack_words_set()
        word_candidate_dic = self.get_candidate_words()

        for sentences in [es_sentence_left, es_sentence_right]:
            for i in tqdm(range(len(sentences))):
                s = sentences[i]
                s = ['/s'] + s + ['/s']
                for j in range(1, len(s)-1):
                    if s[j] in lack_words_set:
                        cand_words = word_candidate_dic.get(s[j])
                        if cand_words == None:
                            continue
                        elif len(cand_words) == 1:
                            s[j] = cand_words[0]
                        else:
                            maxlike_word = None
                            max_proba = 0
                            for cw in cand_words:
                                bigram_word1 = s[j] + '_' + cw
                                bigram_word2 = cw + '_' + s[j]
                                proba = (all_bigram_cnt.get(bigram_word1, 0) + 1) * (all_bigram_cnt.get(bigram_word2, 0) + 1)
                                if proba > max_proba:
                                    max_proba = proba
                                    maxlike_word = cw
                                if proba == max_proba:
                                    if all_unigram_cnt[cw] > all_unigram_cnt[maxlike_word]:
                                        maxlike_word = cw
                            s[j] = maxlike_word
                sentences[i] = s[1:-1]

        if tag == 'train' or tag == 'dev':
            with open(path, 'wb') as pkl:
                 pickle.dump((en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, _), pkl)
            return en_sentence_left, en_sentence_right, es_sentence_left, es_sentence_right, _

        elif tag == 'test_a' or tag == 'test_b':
            with open(path, 'wb') as pkl:
                pickle.dump((es_sentence_left, es_sentence_right), pkl)
            return es_sentence_left, es_sentence_right


if __name__ == '__main__':
    p = Preprocess()
    p.load_train_data('en')
    p.load_train_data('es')
    p.load_test('A')
    p.load_test('B')
    p.load_all_data()
    p.replace_words('train')
    p.replace_words('dev')
    p.replace_words('test_a')
    p.replace_words('test_b')
    p.load_replace_translation_data()
    p.load_all_replace_data()
    p.get_index_data('train')
    p.get_index_data('dev')
    p.get_index_data('test_a')
    p.get_index_data('test_b')
    p.get_index_padding('train')
    p.get_index_padding('dev')
    p.get_index_padding('test_a')
    p.get_index_padding('test_b')
    p.get_length('train')
    p.get_length('dev')
    p.get_length('test_a')
    p.get_length('test_b')
    p.swap_data('train', 'padding')
    p.swap_data('dev', 'padding')


