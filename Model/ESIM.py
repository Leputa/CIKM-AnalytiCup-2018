import tensorflow as tf
from tqdm import tqdm
import os

import sys
sys.path.append('../')

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Model.Embeddings import Embeddings

class ESIM():

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()

        self.lr = 0.0004
        self.keep_rate = 0.5
        self.sentence_length = self.preprocessor.max_length

        self.vec_dim = self.embedding.vec_dim
        self.hidden_dim = 300

        self.batch_size = 128
        self.n_epoch = 20

    def define_model(self):
        self.left_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question')
        self.right_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        #self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix = self.embedding.get_es_embedding_matrix()
            embedding_matrix = tf.Variable(embedding_matrix, trainable=True, name='embedding')
            # embedding_matrix = tf.get_variable('embedding', [self.vocab_size, self.vec_dim], dtype=tf.float32)
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            left__inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.left_sentence),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.left_sentence))
            right_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.right_sentence),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.right_sentence))

            # left__inputs = tf.nn.dropout(left__inputs, self.dropout_keep_prob)
            # right_inputs = tf.nn.dropout(right_inputs, self.dropout_keep_prob)

        # get lengths of unpadded sentences
        # x_seq_length: batch_size * 1
        # x_mask: batch_size * max_length * 1
        left_seq_length, left_mask = tool.length(self.left_sentence)
        right_seq_length, right_mask = tool.length(self.right_sentence)

        # Input Encoding
        with tf.name_scope("BiLSTM"):
            # x_outputs_bi: <batchsize * max_length * hidden_dim>, <batchsize * max_length * hiddden_dim>
            left_outputs_bi, _ = tool.biLSTM(left__inputs, dim=self.hidden_dim, seq_len=left_seq_length, name='left_lstm_cell')
            right_outputs_bi, _ = tool.biLSTM(right_inputs, dim=self.hidden_dim, seq_len=right_seq_length, name='right_lstm_cell')
            # x_outputs: bathsize * max_length * (2*hidden_dim)
            left_outputs = tf.concat(left_outputs_bi, axis=2)
            right_outputs = tf.concat(right_outputs_bi, axis=2)


        # local Inference Modeling
        with tf.name_scope("attention"):
            # scores&scores_mask: batch_size * sentence_length * sentence_length
            scores = tf.matmul(left_outputs, right_outputs, adjoint_b=True)
            scores_mask = tf.einsum('bi,bj->bij', left_mask, right_mask)
            alpha = self.attention_with_mask(scores, right_outputs, 2, scores_mask, name='alpha')
            belta = self.attention_with_mask(scores, left_outputs, 1, scores_mask, name='belta')


    def attention_with_mask(self, scores, output, axis, mask, name=None):
        '''
        求解alpha和belta
        :return: batch_size * sentence_length * vec_dim
        '''
        with tf.name_scope(name):
            target_exp = tf.exp(scores) * mask
            normalize = tf.reduce_sum(target_exp, axis=axis, keepdims=True)
            softmax = target_exp / normalize
            # softmax: batch_size * sentence_length * sentence_length
            # output: batch_size * sentence_length * vec_dim
            alpha_list = []
            for i in range(self.sentence_length):
                if name == 'alpha':
                    softmax_ = tf.expand_dims(softmax[:,i,:], -1)
                elif name == 'belta':
                    softmax_ = tf.expand_dims(softmax[:,:,i], -1)

                alpha_list.append(tf.reduce_sum(
                    tf.multiply(output , softmax_),
                    axis=1,
                    keep_dims=True
                ))

            return tf.concat(alpha_list, axis=1)




if __name__ == '__main__':
    model = ESIM()
    model.define_model()
