import sys
sys.path.append('../')

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,LatentDirichletAllocation,NMF
from sklearn.externals import joblib
from gensim.models import Doc2Vec
import os
import numpy as np
from tqdm import tqdm
import gc
import math
import string
from fuzzywuzzy import fuzz

from Preprocessing import Preprocess
from Config import config
from Model import Embeddings
from Config import tool
from Config.utils import NgramUtil,DistanceUtil,MathUtil




class Feature():
    def __init__(self):
        self.preprocess = Preprocess.Preprocess()
        self.embeddings = Embeddings.Embeddings()


        self.stop_words = []
        stop_words_path = config.data_prefix_path + 'spanish.txt'
        with open(stop_words_path, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                self.stop_words.append(line.strip())

    def del_stop_words(self, corpus):
        stop_corpus = []
        for i in range(len(corpus)):
            sentence = corpus[i]
            corpus[i] = [word for word in sentence if word not in self.stop_words]
        return corpus

    def clean_stop_words(self, left, right, labels, tag):
        stop_left = []
        stop_right = []
        stop_labels = []

        for i in range(len(left)):
            left_list = [word for word in left[i] if word not in self.stop_words]
            right_list = [word for word in right[i] if word not in self.stop_words]

            if tag == 'train' or tag == 'dev':
                if len(left_list) >= 2 and len(right_list) >= 2:
                    stop_left.append(left_list)
                    stop_right.append(right_list)
                    stop_labels.append(labels[i])
            elif tag == 'test':
                stop_left.append(left_list)
                stop_right.append(right_list)

        return stop_left, stop_right, stop_labels


    ### nlp feature
    def count_tf_idf(self, tag='word'):
        print('tfidf model on ' + tag)

        if tag == 'word':
            path = config.cache_prefix_path + 'Tfidf_word_model.m'
        elif tag == 'char':
            path = config.cache_prefix_path + 'Tfidf_char_model.m'

        if os.path.exists(path):
            return joblib.load(path)

        es = self.preprocess.load_replace_translation_data()[0]
        # es = self.del_stop_words(es)

        corpus = [" ".join(sentence) for sentence in es]

        if tag == 'word':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='word',
                ngram_range=(1, 4),
                max_features=20000
            )
        elif tag == 'char':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='char',
                ngram_range=(1, 5),
                max_features=50000
            )
        vectorizer.fit(corpus)

        joblib.dump(vectorizer, path)
        return vectorizer

    def count_nmf(self):
        print('nmf model....')
        path = config.cache_prefix_path + 'nmf_model.m'
        if os.path.exists(path):
            return joblib.load(path)

        vectorizer = self.count_tf_idf()

        es = self.preprocess.load_replace_translation_data()[0]
        corpus = [" ".join(sentence) for sentence in es]
        bow_features = vectorizer.fit_transform(corpus)

        nmf = NMF(n_components=50)
        nmf.fit(bow_features.asfptype())

        joblib.dump(nmf, path)
        return nmf

    def count_lsa(self):
        print('lsa model....')
        path = config.cache_prefix_path + 'lsa_model.m'
        if os.path.exists(path):
            return joblib.load(path)

        vectorizer = self.count_tf_idf()

        es = self.preprocess.load_replace_translation_data()[0]
        corpus = [" ".join(sentence) for sentence in es]
        bow_features = vectorizer.fit_transform(corpus)

        lsa = TruncatedSVD(150, algorithm='arpack')
        lsa.fit(bow_features.asfptype())

        joblib.dump(lsa, path)
        return lsa

    def count_lda(self):
        print('lda model....')
        path = config.cache_prefix_path + 'lda_model.m'
        if os.path.exists(path):
            return joblib.load(path)

        vectorizer = self.count_tf_idf()

        es = self.preprocess.load_replace_translation_data()[0]
        corpus = [" ".join(sentence) for sentence in es]
        bow_features = vectorizer.fit_transform(corpus)

        lda = LatentDirichletAllocation(n_topics=50, learning_method='batch', max_iter=5)
        lda.fit(bow_features.asfptype())

        joblib.dump(lda, path)
        return lda


    def get_tf_idf(self, tag='word'):
        print("Tfidf on " + tag)

        if tag == 'word':
            path = config.cache_prefix_path + 'Tfidf_word.pkl'
        elif tag == 'char':
            path = config.cache_prefix_path + 'Tfidf_char.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        vectorizer = self.count_tf_idf(tag)

        # train
        _, _, train_left, train_right, train_labels = self.preprocess.replace_words('train')
        # train_left, train_right, train_labels = self.clean_stop_words(train_left, train_right, train_labels, 'train')
        train_data = [" ".join(train_left[i]) + " . " + " ".join(train_right[i]) for i in range(len(train_left))]
        train_features = vectorizer.transform(train_data)

        # dev
        _, _, dev_left, dev_right, dev_labels = self.preprocess.replace_words('dev')
        # dev_left, dev_right, dev_labels = self.clean_stop_words(dev_left, dev_right, dev_labels, 'dev')
        dev_data = [" ".join(dev_left[i]) + " . " + " ".join(dev_right[i]) for i in range(len(dev_left))]
        dev_features = vectorizer.transform(dev_data)

        # test
        test_left, test_right = self.preprocess.replace_words('test_b')
        # test_left, test_right, _ = self.clean_stop_words(test_left, test_right, [], 'test')
        assert  len(test_left) == 10000
        test_data = [" ".join(test_left[i]) + " . " + " ".join(test_right[i]) for i in range(len(test_left))]
        test_features = vectorizer.transform(test_data)

        with open(path, 'wb') as pkl:
            pickle.dump(((train_features, train_labels), (dev_features, dev_labels), test_features), pkl)

        return ((train_features, train_labels), (dev_features, dev_labels), test_features)


    def get_doc2vec(self):

        def getVecs(model, start, end, corpus, size):
            vecs = [np.array(model.docvecs[corpus[i].tags[0]]).reshape((1, size)) for i in range(start, end)]
            return np.concatenate(vecs)

        print("getting doc2vec......")

        path = config.cache_prefix_path + 'doc2vec_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        model, dic = self.embeddings.doc2vec()

        train_left_vector = getVecs(model, 0, 20000, dic, self.embeddings.vec_dim)
        train_right_vector = getVecs(model, 20000, 40000, dic, self.embeddings.vec_dim)
        train_vec = np.hstack([train_left_vector, train_right_vector])


        dev_left_vector = getVecs(model, 40000, 41400, dic, self.embeddings.vec_dim)
        dev_right_vector = getVecs(model, 41400, 42800, dic, self.embeddings.vec_dim)
        dev_vec = np.hstack([dev_left_vector, dev_right_vector])

        # test
        test_left_vector = self.getVecs(model, 52800, 62800, dic, self.embeddings.vec_dim)
        test_right_vector = self.getVecs(model, 62800, 72800, dic, self.embeddings.vec_dim)
        test_vec = np.hstack([test_left_vector, test_right_vector])

        with open(path, 'wb') as pkl:
            pickle.dump((train_vec, dev_vec, test_vec), pkl)

        return (train_vec, dev_vec, test_vec)

    def get_average_word2vec(self):
        print ("getting average word2vec")

        path = config.cache_prefix_path + 'word2vec_feature.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        embedding_matrix = self.embeddings.get_embedding_matrix('es')

        # train
        _, _, train_left, train_right, train_labels = self.preprocess.get_index_data('es')
        train_features = np.hstack([self.deal_average_word2vec(train_left, train_right, embedding_matrix)])

        #dev
        dev_left, dev_right, dev_labels = self.preprocess.get_index_data('dev')
        dev_features = np.hstack([self.deal_average_word2vec(dev_left, dev_right, embedding_matrix)])

        # test
        test_left, test_right = self.preprocess.get_index_data('test_b')
        test_features = np.hstack([self.deal_average_word2vec(test_left, test_right, embedding_matrix)])

        with open(path, 'wb') as pkl:
            pickle.dump(((train_features, train_labels), (dev_features, dev_labels), test_features), pkl)

        return ((train_features, train_labels), (dev_features, dev_labels), test_features)


    def deal_average_word2vec(self, left_index, right_index, embedding_matrix):
        left_feature = np.ones((len(left_index), self.embeddings.vec_dim), dtype="float32") * 0.01
        right_feature = np.ones((len(right_index), self.embeddings.vec_dim), dtype="float32") * 0.01

        for i in range(len(left_index)):
            tmp_vec_left, tmp_vec_right = np.zeros(self.embeddings.vec_dim), np.zeros(self.embeddings.vec_dim)

            for index in left_index[i]:
                tmp_vec_left += embedding_matrix[index]
            for index in right_index[i]:
                tmp_vec_right += embedding_matrix[index]

            tmp_vec_left /= len(left_index[i])
            tmp_vec_right /= len(right_index[i])

            left_feature[i] = tmp_vec_left
            right_feature[i] = tmp_vec_right

        return left_feature, right_feature

    def LSA(self, tag = 'word'):
        print("LSA......")

        if tag == 'word':
            path = config.cache_prefix_path + 'word_lsa.pkl'
        elif tag == 'char':
            path = config.cache_prefix_path + 'char_lsa.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        lsa = self.count_lsa()

        ((train_features, train_labels), (dev_features, dev_labels), test_features) = self.get_tf_idf(tag)

        train_features = lsa.transform(train_features.asfptype())
        dev_features = lsa.transform(dev_features.asfptype())
        test_features = lsa.transform(test_features.asfptype())

        with open(path, 'wb') as pkl:
            pickle.dump(((train_features, train_labels), (dev_features, dev_labels), test_features), pkl)

        return ((train_features, train_labels), (dev_features, dev_labels), test_features)


    ### statistics feature
    def load_left_right(self, tag):
        if tag == 'train' or tag == 'dev':
            _, _, left, right, _ = self.preprocess.replace_words(tag)
        elif tag == 'test_a' or tag == 'test_b':
            left, right = self.preprocess.replace_words(tag)

        return left, right


    def get_word_share(self, tag):

        def extract_share_words(left, right):
            q1words = {}
            q2words = {}
            for word in left:
                if word not in self.stop_words:
                    q1words[word] = q1words.get(word, 0) + 1
            for word in right:
                if word not in self.stop_words:
                    q2words[word] = q2words.get(word, 0) + 1
            n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
            n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
            n_tol = sum(q1words.values()) + sum(q2words.values())
            if 1e-6 > n_tol:
                return [0.]
            else:
                return [1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol]

        print('getting share words...')

        path = config.cache_prefix_path + tag + 'share_words.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)

        share_words_list = []
        for i in tqdm(range(len(left))):
            share_words_list.append(extract_share_words(left[i], right[i]))

        assert len(share_words_list) == len(left)

        share_words_list = np.array(share_words_list)
        with open(path, 'wb') as pkl:
            pickle.dump(share_words_list, pkl)

        return share_words_list

    def get_tfidf_sim(self, tag1, tag2):

        def extract_tfidf_sim(left, right):
            left = left.toarray()
            right = right.toarray()
            return [tool.cos_sim(left, right)]

        print("getting tfidf share...")
        path  = config.cache_prefix_path + tag1 + '_' + tag2 + '_tfidf_sim.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        vectorizer = self.count_tf_idf(tag2)

        left, right = self.load_left_right(tag1)
        left = vectorizer.transform([" ".join(sentence) for sentence in left])
        right = vectorizer.transform([" ".join(sentence) for sentence in right])

        tfidf_sim_list = []
        for i in tqdm(range(left.shape[0])):
            tfidf_sim_list.append(extract_tfidf_sim(left[i], right[i]))
            gc.collect()

        assert len(tfidf_sim_list) == left.shape[0]

        tfidf_sim_list = np.array(tfidf_sim_list)
        with open(path, 'wb') as pkl:
            pickle.dump(tfidf_sim_list, pkl)

        return tfidf_sim_list

    def get_word2vec_ave_sim(self, tag):
        def extract_word2vec_ave_sim(left, right):
            return [tool.cos_sim(left, right)]
        print("getting word2vec average sim...")

        path = config.cache_prefix_path + tag + 'word2vec_sim.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        embedding_matrix = self.embeddings.get_embedding_matrix('es')
        if tag == 'train' or tag == 'dev':
            _, _, left, right, _ = self.preprocess.get_index_data(tag)
        elif tag == 'test_b':
            left, right = self.preprocess.get_index_data(tag)
        left, right = self.deal_average_word2vec(left, right, embedding_matrix)

        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_word2vec_ave_sim(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_lda_sim(self, tag):
        # 可能是我参数有问题,这个特征看着不大对劲

        def extract_lda_sim(left, right):
            return [tool.cos_sim(left, right)]

        print("getting lda sim...")
        path = config.cache_prefix_path + tag + '_lda_sim.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        lda = self.count_lda()
        vectorizer = self.count_tf_idf()
        left, right = self.load_left_right(tag)

        left = vectorizer.transform([" ".join(sentence) for sentence in left])
        right = vectorizer.transform([" ".join(sentence) for sentence in right])

        left = lda.transform(left)
        right = lda.transform(right)
        print(left.shape)

        lda_sim_list = []
        for i in tqdm(range(left.shape[0])):
            lda_sim_list.append(extract_lda_sim(left[i], right[i]))
            gc.collect()
        lda_sim_list = np.array(lda_sim_list)

        with open(path, 'wb') as pkl:
            pickle.dump(lda_sim_list, pkl)
        return lda_sim_list


    def get_nmf_sim(self, tag):

        def extract_row(left, right):
            return [tool.cos_sim(left, right)]

        print("getting nmf sim...")
        path = config.cache_prefix_path + tag +'_nmf_sim.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        nmf = self.count_nmf()
        vectorizer = self.count_tf_idf()
        left, right = self.load_left_right(tag)

        left = vectorizer.transform([" ".join(sentence) for sentence in left])
        right = vectorizer.transform([" ".join(sentence) for sentence in right])

        left = nmf.transform(left)
        right = nmf.transform(right)
        print(left.shape)

        feature = []
        for i in tqdm(range(left.shape[0])):
            feature.append(extract_row(left[i], right[i]))
            gc.collect()
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature


    def get_lsa_sim(self, tag):

        def extract_lsa_sim(left, right):
            return [tool.cos_sim(left, right)]

        print("getting lsa sim...")
        path = config.cache_prefix_path + tag + 'lsa_sim.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        lsa = self.count_lsa()
        vectorizer = self.count_tf_idf()
        left, right = self.load_left_right(tag)

        left = vectorizer.transform([" ".join(sentence) for sentence in left])
        right = vectorizer.transform([" ".join(sentence) for sentence in right])

        left = lsa.transform(left)
        right = lsa.transform(right)

        lsa_sim_list = []
        for i in tqdm(range(left.shape[0])):
            lsa_sim_list.append(extract_lsa_sim(left[i], right[i]))
            gc.collect()
        lsa_sim_list = np.array(lsa_sim_list)

        with open(path, 'wb') as pkl:
            pickle.dump(lsa_sim_list, pkl)
        return lsa_sim_list

    def get_doc2vec_sim(self, tag):
        def getVecs(model, start, end, corpus, size):
            vecs = [np.array(model.docvecs[corpus[i].tags[0]]).reshape((1, size)) for i in range(start, end)]
            return np.concatenate(vecs)

        def extract_sim(left, right):
            return [tool.cos_sim(left, right)]

        print("getting doc2vec sim...")

        path  = config.cache_prefix_path + tag + '_doc2vec_sim.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        model, dic = self.embeddings.doc2vec()
        if tag == 'train':
            left_vector = getVecs(model, 0, 20000, dic, self.embeddings.vec_dim)
            right_vector = getVecs(model, 20000, 40000, dic, self.embeddings.vec_dim)
        elif tag == 'dev':
            left_vector = getVecs(model, 40000, 41400, dic, self.embeddings.vec_dim)
            right_vector = getVecs(model, 41400, 42800, dic, self.embeddings.vec_dim)
        elif tag == 'test_b':
            left_vector = getVecs(model, 52800, 62800, dic, self.embeddings.vec_dim)
            right_vector = getVecs(model, 62800, 72800, dic, self.embeddings.vec_dim)

        feature = []
        for i in tqdm(range(left_vector.shape[0])):
            feature.append(extract_sim(left_vector[i], right_vector[i]))
            gc.collect()
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature


    def get_length(self, tag):
        print('getting length...')

        path = config.cache_prefix_path + tag + '_length.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)

        feature = []
        for i in range(len(left)):
            feature.append([len(left[i]), len(right[i])])

        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature



    def get_length_diff(self, tag):
        def extract_row(left, right):
            return [abs(len(left) - len(right))]

        print("getting length diff...")
        path = config.cache_prefix_path + tag + '_length_diff.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)

        len_diff_list = []
        for i in range(len(left)):
            len_diff_list.append(extract_row(left[i], right[i]))

        len_diff_list = np.array(len_diff_list)

        with open(path, 'wb') as pkl:
            pickle.dump(len_diff_list, pkl)
        return len_diff_list


    def get_length_diff_rate(self, tag):
        def extract_row(left, right):
            if max(len(left), len(right)) > 1e-06:
                return [min(len(left), len(right))/max(len(left), len(right))]
            else:
                return [0.]

        print("getting length diff Rate...")
        path = config.cache_prefix_path + tag + '_length_diff_rate.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)

        len_diff_rate_list = []
        for i in range(len(left)):
            len_diff_rate_list.append(extract_row(left[i], right[i]))

        len_diff_rate_list = np.array(len_diff_rate_list)

        with open(path, 'wb') as pkl:
            pickle.dump(len_diff_rate_list, pkl)
        return len_diff_rate_list

    def get_dul_num_sentence(self, tag):
        ############  这个特征感觉没什么用  ##############
        def add_dul_num(dum_num, tag2):
            train_left, train_right = self.load_left_right(tag2)
            for i in range(len(train_left)):
                left = ' '.join(train_left[i])
                right = ' '.join(train_right[i])
                dum_num[left] = dum_num.get(left, 0) + 1
                if left != right:
                    dum_num[right] = dum_num.get(right, 0) + 1
            return dum_num

        def generate_dul_num():
            path = config.cache_prefix_path + 'dul_num_sentence.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as pkl:
                    return pickle.load(pkl)
            dum_num = {}
            dum_num = add_dul_num(dum_num, 'train')
            dum_num = add_dul_num(dum_num, 'dev')
            dum_num = add_dul_num(dum_num, 'test_b')

            with open(path, 'wb') as pkl:
                pickle.dump(dum_num, pkl)

            return dum_num

        print("getting sentence dul num...")
        path = config.cache_prefix_path + tag + '_dum_num.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        dum_num = generate_dul_num()

        left, right = self.load_left_right(tag)

        feature = []
        for i in range(len(left)):
            left_str = ' '.join(left[i])
            right_str = ' '.join(right[i])
            dn1 = dum_num[left_str]
            dn2 = dum_num[right_str]
            feature.append([dn1, dn2, max(dn1, dn2), min(dn1, dn2)])

        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def ngram_jaccard_coef(self, tag):
        def extract_row(q1_words, q2_words):
            fs = list()
            for n in range(1, 4):
                q1_ngrams = NgramUtil.ngrams(q1_words, n)
                q2_ngrams = NgramUtil.ngrams(q2_words, n)
                # jaccard_coef: (A&B) / (A|B)
                fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
            return fs

        print("getting ngram jaccard coef......")
        path = config.cache_prefix_path + tag + '_ngram_jaccard.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)
        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def ngram_dice_distance(self, tag):
        def extract_row(q1_words, q2_words):
            fs = list()
            for n in range(1, 4):
                q1_ngrams = NgramUtil.ngrams(q1_words, n)
                q2_ngrams = NgramUtil.ngrams(q2_words, n)
                # jaccard_coef: (A&B) / (A|B)
                fs.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
            return fs

        print("getting ngram dice distance......")
        path = config.cache_prefix_path + tag + '_ngram_dice.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)
        left, right = self.load_left_right(tag)
        feature = []

        for i in range(len(left)):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_edit_distance(self, tag):
        ##########试了下，这个特征居然没用###########

        def extract_row(q1, q2, distance_func):
            return [distance_func(q1, q2)]

        print("getting distance......")

        path = config.cache_prefix_path + tag + 'edit_dis.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        distance_func = getattr(DistanceUtil, 'edit_dist')

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i], distance_func))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def add_idf_dic(self):
        def add_idf_dic_tag(idf, q_set, length, tag):
            left, right = self.load_left_right(tag)
            length += len(left)
            for i in tqdm(range(len(left))):
                if ' '.join(left[i]) not in q_set:
                    q_set.add(' '.join(left[i]))
                    for word in left[i]:
                        idf[word] = idf.get(word, 0) + 1
                if ' '.join(right[i]) not in q_set:
                    q_set.add(' '.join(right[i]))
                    for word in right[i]:
                        idf[word] = idf.get(word, 0) + 1
            return idf, q_set, length

        path = config.cache_prefix_path + 'idf_dic.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        idf = {}
        q_set = set()
        length = 0

        idf, q_set, length = add_idf_dic_tag(idf, q_set, length, 'train')
        idf, q_set, length = add_idf_dic_tag(idf, q_set, length, 'dev')
        idf, q_set, length = add_idf_dic_tag(idf, q_set, length, 'test')

        for word in idf:
            idf[word] = math.log(length / (idf[word] + 1)) / math.log(2.)

        with open(path, 'wb') as pkl:
            pickle.dump((idf, q_set), pkl)
        return idf, q_set

    def get_idf_dis(self, tag):
        def extract_row(sen1, sen2, idf):
            words = {}
            for word in (sen1 + sen2):
                if word not in words:
                    words[word] = len(words) - 1
            sen_vec_1 = [0 for i in range(len(words))]
            sen_vec_2 = [0 for i in range(len(words))]
            for word in sen1:
                sen_vec_1[words[word]] = idf.get(word, 0)
            for word in sen2:
                sen_vec_2[words[word]] = idf.get(word, 0)
            return [tool.cos_sim(np.array(sen_vec_1), np.array(sen_vec_2))]

        print('getting tfidf distance......')
        path = config.cache_prefix_path + tag + '_idf_dis.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        idf, _ = self.add_idf_dic()
        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i], idf))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_tfidf_word_share(self, tag):
        def extract_row(q1_words, q2_words, idf):
            q1words = {}
            q2words = {}
            for word in q1_words:
                q1words[word] = q1words.get(word, 0) + 1
            for word in q2_words:
                q2words[word] = q2words.get(word, 0) + 1
            sum_shared_word_in_q1 = sum([q1words[w] * idf.get(w, 0) for w in q1words if w in q2words])
            sum_shared_word_in_q2 = sum([q2words[w] * idf.get(w, 0) for w in q2words if w in q1words])
            sum_tol = sum([q1words[w] * idf.get(w,0) for w in q1words]) + sum([q2words[w] * idf.get(w,0) for w in q2words])
            if 1e-6 > sum_tol:
                return [0.]
            else:
                return [1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol]

        print("getting idf words match share......")

        path = config.cache_prefix_path + tag + '_idf_words_share.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        idf, _ = self.add_idf_dic()
        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i], idf))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_tfidf_statistics(self, tag):
        def extract_row(q1, q2, tfidf):
            fs = []
            fs.append(np.sum(tfidf.transform([str(q1)]).data))
            fs.append(np.sum(tfidf.transform([str(q2)]).data))
            fs.append(np.mean(tfidf.transform([str(q1)]).data))
            fs.append(np.mean(tfidf.transform([str(q2)]).data))
            fs.append(len(tfidf.transform([str(q1)]).data))
            fs.append(len(tfidf.transform([str(q2)]).data))
            return fs

        print("getting tfidf statistics feature......")
        path = config.cache_prefix_path + tag + '_tfidf_statistics.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        vectorizer = self.count_tf_idf()
        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            left_s = ' '.join(left[i])
            right_s = ' '.join(right[i])
            feature.append(extract_row(left_s, right_s, vectorizer))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_no_feature(self, tag):
        # 加了这个特征居然没有任何区别，感觉哪儿有bug......
        def extract_row(q1_words, q2_words):
            not_cnt1 = q1_words.count(b'no')
            not_cnt2 = q2_words.count(b'no')
            not_cnt1 += q1_words.count(b'ni')
            not_cnt2 += q2_words.count(b'ni')
            not_cnt1 += q1_words.count(b'nunca')
            not_cnt2 += q2_words.count(b'nunca')

            fs = list()
            fs.append(not_cnt1)
            fs.append(not_cnt2)
            if not_cnt1 > 0 and not_cnt2 > 0:
                fs.append(1.)
            else:
                fs.append(0.)
            if (not_cnt1 > 0) or (not_cnt2 > 0):
                fs.append(1.)
            else:
                fs.append(0.)
            if not_cnt2 <= 0 < not_cnt1 or not_cnt1 <= 0 < not_cnt2:
                fs.append(1.)
            else:
                fs.append(0.)
            return fs

        print("getting not word feature......")
        path = config.cache_prefix_path + tag + '_not_word.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_fuzz_feature(self, tag):
        def extract_row(q1_words, q2_words):
            fs = []
            fs.append(fuzz.QRatio(set(q1_words), set(q2_words)))
            fs.append(fuzz.WRatio(set(q1_words), set(q2_words)))
            fs.append(fuzz.partial_ratio(set(q1_words), set(q2_words)))
            fs.append(fuzz.partial_token_set_ratio(set(q1_words), set(q2_words)))
            fs.append(fuzz.partial_token_sort_ratio(set(q1_words), set(q2_words)))
            fs.append(fuzz.token_set_ratio(set(q1_words), set(q2_words)))
            fs.append(fuzz.token_sort_ratio(set(q1_words), set(q2_words)))
            return fs

        print("getting fuzz feature......")
        path = config.cache_prefix_path + tag + '_fuzz.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_longest_common_sequence(self, tag):
        def extract_row(q1_words, q2_words):
            len1 = len(q1_words)
            len2 = len(q2_words)
            dp = [[0 for j in range(len2)] for i in range(len1)]
            for i in range(len1):
                for j in range(len2):
                    if q1_words[i] == q2_words[j]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            if len1 + len2 < 1e-6:
                return [0.]
            else:
                return [dp[len1 - 1][len2 - 1] * 1.0 / (len1 + len2)]

        print("getting longest common sequence feature......")
        path = config.cache_prefix_path + tag + '_common_sequence.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_longest_common_prefix_suffix(self, tag):
        # 这个特征好像没啥用
        def extract_row(q1_words, q2_words):
            len1 = len(q1_words)
            len2 = len(q2_words)
            if len1 + len2 < 1e-06:
                return [0., 0.]

            fs = []
            max_prefix = 0
            min_len = min(len1, len2)
            for i in range(min_len):
                if q1_words[i] == q2_words[i]:
                    max_prefix += 1
            fs.append(max_prefix * 1.0 / (len1 + len2))

            q1_words.reverse()
            q2_words.reverse()
            max_prefix = 0
            for i in range(min_len):
                if q1_words[i] == q2_words[i]:
                    max_prefix += 1
            fs.append(max_prefix * 1.0 / (len1 + len2))
            return fs

        print("getting longest common prefix&suffix feature......")
        path = config.cache_prefix_path + tag + '_common_prefix_suffix.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_lcs_diff(self, tag):
        # 这个特征也没用
        def extract_row(sen1, sen2):
            len1 = len(sen1)
            len2 = len(sen2)

            dp = [[0 for j in range(len2)] for i in range(len1)]
            offset = 0
            for i in range(1, len1):
                for j in range(1, len2):
                    if sen1[i] == sen2[j]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        of = abs(j - i)
                        offset += of
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return [offset * 1.0 / (len1 + len2)]

        print("getting lsc diff feature......")
        path = config.cache_prefix_path + tag + '_lsc_diff.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_inter_pos(self, tag):
        # 这个特征没啥用
        def extract_row(q1, q2):
            mode = ["mean", "std", "max", "min"]
            pos_list = [abs(i - q1.index(o)) for i,o in enumerate(q2, start=1) if o in q1]
            if len(pos_list) == 0:
                pos_list = [0]
            return MathUtil.aggregate(pos_list, mode)

        print("getting intersecter position feature......")
        path = config.cache_prefix_path + tag + '_inter_pos.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature

    def get_w2v_feature(self, tag):
        # 这个特征居然没用
        def extract_row(q1, q2, word2index, embedding_matrix):
            common_words = set(q1).intersection(set(q2))
            q1 = [word for word in q1 if word not in common_words]
            q2 = [word for word in q2 if word not in common_words]
            mode = ["mean", "std", "max", "min"]
            sims = []
            dists = []
            for i in range(len(q1)):
                for j in range(len(q2)):
                    v1 = embedding_matrix[word2index[q1[i]]]
                    v2 = embedding_matrix[word2index[q2[j]]]
                    sims.append(tool.cos_sim(v1, v2))
                    vec_diff = v1 - v2
                    dist = np.sqrt(np.sum(vec_diff**2))
                    dists.append(dist)
            sims_feature = MathUtil.aggregate(sims, mode)
            dists_feature = MathUtil.aggregate(dists, mode)
            sims_feature.extend(dists_feature)
            return sims_feature

        print("getting w2v feature......")
        path = config.cache_prefix_path + tag + '_w2v_sim_dis.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        embedding_matrix = self.embeddings.get_es_embedding_matrix()
        word2index = self.preprocess.es2index('es')
        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(left[i], right[i], word2index, embedding_matrix))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature


    def get_same_subgraph_feature(self, tag):
        def sentenceInSetByPeopelGraphResult(sen, graph_result):
            for sub_graph in graph_result:
                flag = 0
                for ansSection in sub_graph:
                    for orWord in sub_graph:
                        if orWord in sen:
                            flag += 1
                            break
                if flag == len(sub_graph):
                    return True
            return False

        def bothSentencesInSameSubGraph(graph_result, sen1, sen2):
            if sentenceInSetByPeopelGraphResult(sen1, graph_result) and sentenceInSetByPeopelGraphResult(sen2, graph_result):
                return True
            return False

        def singleSentencesInSameSubGraph(graph_result, sen1, sen2):
            if (sentenceInSetByPeopelGraphResult(sen1, graph_result) and not sentenceInSetByPeopelGraphResult(sen2, graph_result)) or (
                not sentenceInSetByPeopelGraphResult(sen1, graph_result) and sentenceInSetByPeopelGraphResult(sen2, graph_result)):
                return True
            return False

        def noneSentencesInSameSubGraph(graph_result, sen1, sen2):
            if sentenceInSetByPeopelGraphResult(sen1, graph_result) or sentenceInSetByPeopelGraphResult(sen2, graph_result):
                return False
            return True

        def extract_row(graph_result, sen1, sen2):
            bothInSameSubGraph = 0
            singleInSameSubGraph = 0
            noneInSameSubGraph = 0

            if bothSentencesInSameSubGraph(graph_result, sen1, sen2):
                bothInSameSubGraph = 1
            if singleSentencesInSameSubGraph(graph_result, sen1, sen2):
                singleInSameSubGraph = 1
            if noneSentencesInSameSubGraph(graph_result, sen1, sen2):
                noneInSameSubGraph = 1
            fs = []
            fs.append(bothInSameSubGraph)
            fs.append(singleInSameSubGraph)
            fs.append(noneInSameSubGraph)
            return fs


        print("getting same subgraph feature")
        path = config.cache_prefix_path + tag + "_subgraph.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        graph_result = [[['impuesto']],
                        [['cómo'], ['reporto', 'enviar', 'informar', 'reportar', 'informo'], ['proveedor']],
                        [['hacer', 'cómo'], ['pedido']], [['bancaria']],
                        [['Quiero'], ['pagar']], [['no', 'ni', 'nunca'], ['pedido']], [['Donde'], ['cupones']],
                        [['número'], ['teléfono']], [['recibir'], ['pedido']],
                        [['recibir', 'recibir'], ['no', 'ni', 'nunca']], [['confiable'], ['vendedor', 'proveedor']],
                        [['protección'], ['comprador', 'compra']], [['mi'], ['preguntar']]]

        left, right = self.load_left_right(tag)
        feature = []

        for i in tqdm(range(len(left))):
            feature.append(extract_row(graph_result, left[i], right[i]))

        feature = np.array(feature)
        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature


    def addtional_feature(self, tag, modeltype):

        lsa_sim = self.get_lsa_sim(tag)
        # lda_sim = self.get_lda_sim(tag)
        word_share =  self.get_word_share(tag)
        doc2vec_sim = self.get_doc2vec_sim(tag)

        # ABCNN只选了这3个
        if modeltype.startswith('ABCNN') or modeltype == 'LexDecomp':
            return np.hstack([lsa_sim, word_share, doc2vec_sim])

        word2vec_sim = self.get_word2vec_ave_sim(tag)
        tfidf_char_sim = self.get_tfidf_sim(tag, 'char')
        length = self.get_length(tag)
        length_diff = self.get_length_diff(tag)
        length_diff_rate = self.get_length_diff_rate(tag)

        ngram_jaccard_dis = self.ngram_jaccard_coef(tag)
        ngram_dice_dis = self.ngram_dice_distance(tag)
        # idf_word_share = self.get_tfidf_word_share(tag)      # 这两个特征线上线下不一致

        # not_words_count = self.get_no_feature(tag)
        edit_dictance = self.get_edit_distance(tag)
        fuzz = self.get_fuzz_feature(tag)
        common_sequence = self.get_longest_common_sequence(tag)
        # prefix_suffix = self.get_longest_common_prefix_suffix(tag)
        # lsc_diff = self.get_lcs_diff(tag)
        # inter_pos = self.get_inter_pos(tag)
        # w2v_sim_dist = self.get_w2v_feature(tag)
        # tfidf_word_sim = self.get_tfidf_sim(tag, 'word')  # 这个特征貌似也没用了
        # sub_graph = self.get_same_subgraph_feature(tag)
        # tfidf_statistics = self.get_tfidf_statistics(tag)    # 这两个特征线上线下不一致
        # idf_word_share = self.get_tfidf_word_share(tag)      # 这两个特征线上线下不一致
        # idf_dis = self.get_idf_dis(tag)

        if modeltype == 'Xgboost' or modeltype == 'LightGbm' or modeltype == 'FM_FTRL':
            return np.hstack([lsa_sim, tfidf_char_sim, word_share, doc2vec_sim, word2vec_sim, length, length_diff, length_diff_rate, \
                            ngram_jaccard_dis, ngram_dice_dis, fuzz, common_sequence,]) #


if __name__ == '__main__':
    feature = Feature()
    feature.get_tfidf_sim('train','char')
    feature.get_tfidf_word_share('train')
    feature.get_tfidf_word_share('dev')
    feature.get_tfidf_word_share('test_b')
    feature.get_tfidf_statistics('train')
    feature.get_tfidf_statistics('dev')
    feature.get_tfidf_statistics('test_b')
    feature.get_no_feature('train')
    feature.get_no_feature('dev')
    feature.get_no_feature('test_b')
    feature.get_fuzz_feature('train')
    feature.get_fuzz_feature('dev')
    feature.get_fuzz_feature('test_b')
    feature.get_longest_common_sequence('train')
    feature.get_longest_common_sequence('dev')
    feature.get_longest_common_sequence('test_b')
    feature.get_longest_common_prefix_suffix('train')
    feature.get_longest_common_prefix_suffix('dev')
    feature.get_longest_common_prefix_suffix('test_b')
    feature.get_lcs_diff('train')
    feature.get_lcs_diff('dev')
    feature.get_lcs_diff('test_b')
    feature.get_inter_pos('train')
    feature.get_inter_pos('dev')
    feature.get_inter_pos('test_b')
    feature.get_w2v_feature('train')
    feature.get_w2v_feature('dev')
    feature.get_w2v_feature('test_b')
    feature.get_nmf_sim('train')
    feature.get_nmf_sim('dev')
    feature.get_nmf_sim('test_b')
    feature.get_same_subgraph_feature('train')
    feature.get_same_subgraph_feature('dev')
    feature.get_same_subgraph_feature('test_b')
