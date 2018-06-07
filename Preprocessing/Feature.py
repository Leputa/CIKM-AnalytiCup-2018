import sys
sys.path.append('../')

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os


from Preprocessing import Preprocess
from Config import config


class Feature():
    def __init__(self):
        self.preprocess = Preprocess.Preprocess()

    def word_tf_idf(self, tag='word'):
        print("Tfidf on " + tag)

        if tag == 'word':
            path = config.cache_prefix_path + 'Tfidf_word.pkl'
        elif tag == 'char':
            path = config.cache_prefix_path + 'Tfidf_char.pkl'

        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        es = self.preprocess.load_all_data()[0]
        corpus = [" ".join(sentence) for sentence in es]

        if tag == 'word':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='word',
                ngram_range=(1, 3),
                max_features=10000
            )
        elif tag == 'char':
            vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='char',
                ngram_range=(1, 4),
                max_features=30000
            )
        vectorizer.fit(corpus)

        # train
        _, _, train_left, train_right, train_labels = self.preprocess.load_train_data('en')
        train_data = [" ".join(train_left[i]) + " . " + " ".join(train_right[i]) for i in range(len(train_left))]
        train_features = vectorizer.transform(train_data)

        # dev
        _, _, dev_left, dev_right, dev_labels = self.preprocess.load_train_data('es')
        dev_data = [" ".join(dev_left[i]) + " . " + " ".join(dev_right[i]) for i in range(len(dev_left))]
        dev_features = vectorizer.transform(dev_data)

        # test
        test_left, test_right = self.preprocess.load_test()
        test_data = [" ".join(test_left[i]) + " . " + " ".join(test_right[i]) for i in range(len(test_left))]
        test_features = vectorizer.transform(test_data)

        with open(path, 'wb') as pkl:
            pickle.dump(((train_features, train_labels), (dev_features, dev_labels), test_features), pkl)

        return ((train_features, train_labels), (dev_features, dev_labels), test_features)


if __name__ == '__main__':
    feature = Feature()
    feature.word_tf_idf('char')
