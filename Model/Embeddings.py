import pickle
import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')

from Preprocessing import Preprocess

from Config import config
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument  


class Embeddings():
    def __init__(self):
        self.scale = 0.1
        self.vec_dim = 300
        self.preprocessor = Preprocess.Preprocess()

    def word2vec(self, word_emb, word, scale, vec_dim):
        unknown_word = np.random.uniform(-scale,scale,vec_dim)
        if word in word_emb:
            res = word_emb[word]
            flag = 0
        else:
            res = unknown_word
            flag = 1
        return res,flag

    def get_es_embedding_matrix(self):
        print("ES Embedding!")

        path = config.cache_prefix_path + "es_index2vec.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as pkl:
                return pickle.load(pkl)

        word_emb = KeyedVectors.load_word2vec_format(config.ES_EMBEDDING_MATRIX, encoding='utf-8')
        word2index = self.preprocessor.es2index()

        vocal_size = len(word2index)
        index2vec = np.ones((vocal_size, self.vec_dim), dtype="float32") * 0.01
        unk_count = 0

        for word in word2index:
            index = word2index[word]
            if index == 0:
                continue
            vec, flag = self.word2vec(word_emb, word, self.scale, self.vec_dim)
            index2vec[index] = vec
            unk_count += flag

        print("emb vocab size: ", len(word_emb.vocab))
        print("unknown words count: ", unk_count)
        print("index2vec size: ", len(index2vec))

        with open(path, 'wb') as pkl:
            pickle.dump(index2vec, pkl)
        return index2vec

    def doc2vec(self):
        print("doc2vec...")
        path = config.cache_prefix_path + 'doc2vec.embedding'

        if os.path.exists(path):
            return Doc2Vec.load(path)

        es = self.preprocessor.load_all_data()[0]
        model = Doc2Vec(TaggedLineDocument(inp), 
                        size = 100,
                        window = 5,
                        min_count = 1,
                        sample = 1e-3,
                        negative = 5,
                        workers = 4)
        model.train(es, total_examples=model.corpus_count, epochs=50)
        model.save(path)

        return model


if __name__ == '__main__':
    embedding = Embeddings()
    embedding.get_es_embedding_matrix()

