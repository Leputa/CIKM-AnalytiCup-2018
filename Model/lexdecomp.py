import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import gc
from tensorflow.python import debug as tf_debug

import sys
sys.path.append("../")

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Model.Embeddings import Embeddings
from Preprocessing import Feature

class LexDecomp():
    # implementation of the Answer Selection (AS) model proposed in the paper
    # Sentence Similarity Learning by Lexical Decomposition and Composition, by (Wang et al., 2016).
    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()
        self.Feature = Feature.Feature()

        self.lr = 0.002
        self.batch_size = 64
        self.n_epoch = 20

        self.sentence_length = self.preprocessor.max_length
        self.w = 3

        self.eclipse = 1e-10


    def define_model(self):
        self.question = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question')
        self.answer = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        #self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix = self.embedding.get_es_embedding_matrix()
            embedding_matrix = tf.Variable(embedding_matrix, trainable=True, name='embedding')
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            left_inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.question),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.question))
            right_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.answer),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.answer))

        with tf.name_scope('sematic_matching'):
            A = self.make_word_sim(left_inputs, right_inputs)
            left_hat = self.semantic_match(right_inputs, A, self.w, scope='left', type='local_w')
            right_hat = self.semantic_match(left_inputs, tf.transpose(A, [0, 2, 1]), self.w, scope='right', type='local_w')

        with tf.name_scope('Decomposition'):
            left_pos, left_neg = self.decompose(left_inputs, left_hat, scope='left', type='linear')
            right_pos, right_neg = self.decompose(right_inputs, right_hat, scope='right', type='linear')



    def decompose(self, X, X_hat, scope, type='linear'):
        with tf.variable_scope('decompose_'+scope):
            if type == 'rigid':
                mask_pos = tf.cast(tf.equal(X, X_hat), dtype='float32')
                X_pos = X * mask_pos
                mask_neg = tf.cast(tf.equal(mask_pos, 0.0), dtype='float32')
                X_neg = X * mask_neg
            elif type == 'linear':
                denom = (tf.linalg.norm(X, axis=-1, keepdims=True)) * (tf.linalg.norm(X_hat, axis=-1, keepdims=True))
                alpha = tf.reduce_sum(X * X_hat, axis=-1, keepdims=True) / denom
                X_pos = alpha * X
                X_neg = (1 - alpha) * X
            elif type == 'orthogonal':
                denom = tf.reduce_sum(X_hat * X_hat, axis=-1, keepdims=True)
                X_pos = tf.reduce_sum(X * X_hat, axis=-1, keepdims=True) / denom  * X_hat
                X_neg = X - X_pos
            return X_pos, X_neg




    def semantic_match(self, X, A, w, scope, type='local_w'):
        # x: batch * length2 * vec_dim
        # A: batch * length1 * length2
        # return: batch * length * vec_dim
        # type == 'max'只需将w设置为0， 'global'只需将w设置为sentence_length
        with tf.variable_scope('semantic_' + scope):
            pivot = tf.argmax(A, axis=-1)
            if type == 'global':
                return tf.matmul(A, X)/tf.expand_dims(tf.reduce_sum(A, axis=-1), axis=-1)
            elif type == 'local_w':
                upper = tf.expand_dims(tf.minimum(pivot + w, self.sentence_length),-1)
                lower = tf.expand_dims(tf.maximum(pivot - w, 0), -1)
                indices =  tf.tile(tf.expand_dims(tf.expand_dims(tf.range(start=0, limit=self.sentence_length, dtype='int64'), axis=0), axis=0), [self.batch_size, 50, 1])
                mask = tf.cast(((indices>=lower)&(indices<=upper)), 'float32')
                return tf.matmul(A*mask, X)/tf.expand_dims(tf.reduce_sum(A*mask, axis=-1), axis=-1)

    def make_word_sim(self, x, y):
        dot = tf.einsum('abd,acd->abc', x, y)
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(x), axis=2, keepdims=True))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(y), axis=2, keepdims=True))
        return dot / tf.einsum('abd,acd->abc', norm1, norm2)


if __name__ == '__main__':
    model = LexDecomp()
    a = model.define_model()


