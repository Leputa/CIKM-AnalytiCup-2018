import sys
sys.path.append('../')

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from gensim.models import Doc2Vec
import os
import numpy as np
from tqdm import tqdm
import gc

from Preprocessing import Preprocess
from Config import config
from Model import Embeddings
from Config import tool
from Config.utils import  NgramUtil,DistanceUtil




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

        es = self.preprocess.load_all_data()[0]
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

    def count_lsa(self):
        print('lsa model....')
        path = config.cache_prefix_path + 'lsa_model.m'
        if os.path.exists(path):
            return joblib.load(path)

        vectorizer = self.count_tf_idf()

        es = self.preprocess.load_all_data()[0]
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

        es = self.preprocess.load_all_data()[0]
        corpus = [" ".join(sentence) for sentence in es]
        bow_features = vectorizer.fit_transform(corpus)

        lda = LatentDirichletAllocation(n_topics=10, learning_method='batch', max_iter=30)
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
        _, _, train_left, train_right, train_labels = self.preprocess.load_train_data('en')
        # train_left, train_right, train_labels = self.clean_stop_words(train_left, train_right, train_labels, 'train')
        train_data = [" ".join(train_left[i]) + " . " + " ".join(train_right[i]) for i in range(len(train_left))]
        train_features = vectorizer.transform(train_data)

        # dev
        _, _, dev_left, dev_right, dev_labels = self.preprocess.load_train_data('es')
        # dev_left, dev_right, dev_labels = self.clean_stop_words(dev_left, dev_right, dev_labels, 'dev')
        dev_data = [" ".join(dev_left[i]) + " . " + " ".join(dev_right[i]) for i in range(len(dev_left))]
        dev_features = vectorizer.transform(dev_data)

        # test
        test_left, test_right = self.preprocess.load_test()
        # test_left, test_right, _ = self.clean_stop_words(test_left, test_right, [], 'test')
        # assert  len(test_left) == 5000
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
        test_left_vector = self.getVecs(model, 42800, 47800, dic, self.embeddings.vec_dim)
        test_right_vector = self.getVecs(model, 47800, 52800, dic, self.embeddings.vec_dim)
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


        embedding_matrix = self.embeddings.get_es_embedding_matrix()

        # train
        train_left, train_right, train_labels = self.preprocess.get_es_index_data('train')
        train_features = np.hstack([self.deal_average_word2vec(train_left, train_right, embedding_matrix)])

        #dev
        dev_left, dev_right, dev_labels = self.preprocess.get_es_index_data('dev')
        dev_features = np.hstack([self.deal_average_word2vec(dev_left, dev_right, embedding_matrix)])

        # test
        test_left, test_right = self.preprocess.get_es_index_data('test')
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
        dic = {
            'train':'en',
            'dev': 'es'
        }
        if tag == 'train' or tag == 'dev':
            _, _, left, right, _ = self.preprocess.load_train_data(dic[tag])
        elif tag == 'test':
            left, right = self.preprocess.load_test()

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

        if tag == 'train':
            path = config.cache_prefix_path + 'share_words_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'share_words_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'share_words_test.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)
        dic = {
            'train':'en',
            'dev': 'es'
        }

        if tag == 'train' or tag == 'dev':
            _, _, left, right, _ = self.preprocess.load_train_data(dic[tag])
        elif tag == 'test':
            left, right = self.preprocess.load_test()

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
        if tag1 == 'train':
            path = config.cache_prefix_path + tag2 +'_tfidf_sim_train.pkl'
        elif tag1 == 'dev':
            path = config.cache_prefix_path + tag2 + '_tfidf_sim_dev.pkl'
        elif tag1 == 'test':
            path = config.cache_prefix_path + tag2 + '_tfidf_sim_test.pkl'
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
        if tag == 'train':
            path = config.cache_prefix_path + 'word2vec_sim_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'word2vec_sim_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'word2vec_sim_test.pkl'
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        embedding_matrix = self.embeddings.get_es_embedding_matrix()
        if tag == 'train' or tag == 'dev':
            left, right, _ = self.preprocess.get_es_index_data(tag)
        elif tag == 'test':
            left, right = self.preprocess.get_es_index_data(tag)
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
        if tag == 'train':
            path = config.cache_prefix_path + 'lda_sim_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'lda_sim_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'lda_sim_test.pkl'

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


    def get_lsa_sim(self, tag):

        def extract_lsa_sim(left, right):
            return [tool.cos_sim(left, right)]

        print("getting lsa sim...")
        if tag == 'train':
            path = config.cache_prefix_path + 'lsa_sim_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'lsa_sim_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'lsa_sim_test.pkl'
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
        if tag == 'train':
            path = config.cache_prefix_path + 'doc2vec_sim_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'doc2vec_sim_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'doc2vec_sim_test.pkl'
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
        elif tag == 'test':
            left_vector = getVecs(model, 42800, 47800, dic, self.embeddings.vec_dim)
            right_vector = getVecs(model, 47800, 52800, dic, self.embeddings.vec_dim)

        feature = []
        for i in tqdm(range(left_vector.shape[0])):
            feature.append(extract_sim(left_vector[i], right_vector[i]))
            gc.collect()
        feature = np.array(feature)

        with open(path, 'wb') as pkl:
            pickle.dump(feature, pkl)
        return feature


    def get_length(self, tag):
        print('getting length..')

        if tag == 'train':
            path = config.cache_prefix_path + 'length_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'length_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'length_test.pkl'
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
        if tag == 'train':
            path = config.cache_prefix_path + 'length_diff_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'length_diff_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'length_diff_test.pkl'
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
        if tag == 'train':
            path = config.cache_prefix_path + 'length_diff_rate_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'length_diff_rate_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'length_diff_rate_test.pkl'
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
            dum_num = add_dul_num(dum_num, 'test')

            with open(path, 'wb') as pkl:
                pickle.dump(dum_num, pkl)

            return dum_num

        print("getting sentence dul num...")
        if tag == 'train':
            path = config.cache_prefix_path + 'dum_num_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'dum_num_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'dum_num_test.pkl'
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
        if tag == 'train':
            path = config.cache_prefix_path + 'ngram_jaccard_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'ngram_jaccard_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'ngram_jaccard_test.pkl'
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
        if tag == 'train':
            path = config.cache_prefix_path + 'ngram_dice_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'ngram_dice_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'ngram_dice_test.pkl'
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

        if tag == 'train':
            path = config.cache_prefix_path + 'edit_dis_train.pkl'
        elif tag == 'dev':
            path = config.cache_prefix_path + 'edit_dis_dev.pkl'
        elif tag == 'test':
            path = config.cache_prefix_path + 'edit_dis_test.pkl'
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

    def addtional_feature(self, tag):
        # ABCNN只选了这3个
        lsa_sim = self.get_lsa_sim(tag)
        #lda_sim = self.get_lda_sim(tag)
        tfidf_char_sim = self.get_tfidf_sim(tag, 'char')
        word_share =  self.get_word_share(tag)
        doc2vec_sim = self.get_doc2vec_sim(tag)
        word2vec_sim = self.get_word2vec_ave_sim(tag)

        # return np.hstack([lsa_sim, word_share, doc2vec_sim])


        length = self.get_length(tag)
        length_diff = self.get_length_diff(tag)
        length_diff_rate = self.get_length_diff_rate(tag)

        ngram_jaccard_dis = self.ngram_jaccard_coef(tag)
        ngram_dice_dis = self.ngram_dice_distance(tag)

        #edit_dictance = self.get_edit_distance(tag)


        return np.hstack([lsa_sim, tfidf_char_sim, word_share, doc2vec_sim, word2vec_sim, length, length_diff, length_diff_rate, ngram_jaccard_dis, ngram_dice_dis])





if __name__ == '__main__':
    feature = Feature()
    feature.get_tfidf_sim('train','char')
