import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np

import sys
sys.path.append('../')

from Config import config
from Config import tool
from Model.BaseDeepModel import BaseDeepModel

class ESIM(BaseDeepModel):
    def __init__(self):
        super().__init__()

        self.model_type = 'ESIM'
        self.lr = 0.0001
        self.hidden_dim = 300

        self.batch_size = 128
        self.n_epoch = 20

        self.keep_prob = 1.0
        self.l2_reg = 0.0004


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
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            left_inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.left_sentence),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.left_sentence))
            right_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.right_sentence),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.right_sentence))

            left_inputs = tf.nn.dropout(left_inputs, self.dropout_keep_prob)
            right_inputs = tf.nn.dropout(right_inputs, self.dropout_keep_prob)

        # get lengths of unpadded sentences
        # x_seq_length: batch_size * 1
        # x_mask: batch_size * max_length * 1
        left_seq_length, left_mask = tool.length(self.left_sentence)
        right_seq_length, right_mask = tool.length(self.right_sentence)

        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
            cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            return cell_dropout

        # Input Encoding
        with tf.name_scope("LSTM"):
            with tf.variable_scope('left_lstm'):
                lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
                left_outputs, _ = tf.nn.dynamic_rnn(lstm_fwd, left_inputs, sequence_length=left_seq_length, dtype=tf.float32)
            with tf.variable_scope('right_lstm'):
                lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
                right_outputs, _ = tf.nn.dynamic_rnn(lstm_bwd, right_inputs, sequence_length=right_seq_length, dtype=tf.float32)


        # 3.2 Local Inference Modeling
        with tf.name_scope("attention"):
            # scores&scores_mask: batch_size * sentence_length * sentence_length
            scores = tf.matmul(left_outputs, right_outputs, adjoint_b=True)
            scores_mask = tf.einsum('bi,bj->bij', left_mask, right_mask)
            alpha = self.attention_with_mask(scores, right_outputs, 2, scores_mask, name='alpha')
            beta = self.attention_with_mask(scores, left_outputs, 1, scores_mask, name='beta')

        with tf.name_scope('Enhancement'):
            m_a = tf.concat([left_outputs, alpha, left_outputs-alpha, left_outputs*alpha], axis=2)
            m_b = tf.concat([right_outputs, beta, right_outputs-beta, right_outputs*beta], axis=2)

        # Inference Composition
        with tf.name_scope('composition_layer'):
            with tf.variable_scope('left_lstm_v'):
                lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
                v_left_output, _ = tf.nn.dynamic_rnn(lstm_fwd, m_a, sequence_length=left_seq_length, dtype=tf.float32)
            with tf.variable_scope('right_lstm_v'):
                lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
                v_right_output, _ = tf.nn.dynamic_rnn(lstm_bwd, m_b, sequence_length=left_seq_length, dtype=tf.float32)

        with tf.name_scope('pooling'):
            with tf.name_scope('ave_pooling'):
                v_left_ave =tf.div(tf.reduce_sum(v_left_output, axis=1), tf.expand_dims(tf.cast(left_seq_length, tf.float32), -1))
                v_right_ave = tf.div(tf.reduce_sum(v_right_output, axis=1), tf.expand_dims(tf.cast(right_seq_length, tf.float32), -1))
            with tf.name_scope('max_pooling'):
                v_left_max = tf.reduce_max(v_left_output, axis=1)
                v_right_max = tf.reduce_max(v_right_output, axis=1)
            v = tf.concat([v_left_ave, v_right_ave, v_left_max, v_left_max], axis=1)


        with tf.name_scope('MLP'):
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.W_mlp = tf.get_variable(shape=(self.hidden_dim*4, self.hidden_dim), initializer=xavier_init, name='W_mlp')
            self.b_mlp = tf.get_variable(shape=(self.hidden_dim), initializer=xavier_init, name='b_mlp')
            h_mlp = tf.nn.tanh(tf.matmul(v, self.W_mlp) + self.b_mlp)

            h_bn = tf.layers.batch_normalization(h_mlp, name='hidden_bn')
            h_drop = tf.nn.dropout(h_bn, self.dropout_keep_prob, name='hidden_drop')

        with tf.name_scope('output'):
            self.W_cl = tf.get_variable(shape=(self.hidden_dim, self.num_classes), initializer=xavier_init, name='W_cl')
            self.b_cl = tf.get_variable(shape=(self.num_classes), initializer=xavier_init, name='b_cl')
            self.output = tf.matmul(h_drop, self.W_cl) + self.b_cl

            self.prediction = tf.contrib.layers.softmax(self.output)[:, 1]

        with tf.name_scope('regularizer'):
            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)
            tf.add_to_collection('losses', regularizer(self.W_mlp))
            tf.add_to_collection('losses', regularizer(self.W_cl))


        with tf.name_scope('cost'):
            self.cost = tf.add(
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label)),
                tf.add_n(tf.get_collection('losses'))
            )
            self.cost_non_reg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label))

        with tf.name_scope('acc'):
            correct = tf.nn.in_top_k(self.output, self.label, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def attention_with_mask(self, scores, output, axis, mask, name=None):
        '''
        求解alpha和belta
        :return: batch_size * sentence_length * vec_dim
        '''
        with tf.name_scope(name):
            max_axis = tf.reduce_max(scores, axis=axis, keep_dims=True)
            target_exp = tf.exp(scores - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis=axis, keep_dims=True)
            softmax = target_exp / (normalize + self.eclipse)
            # softmax: batch_size * sentence_length * sentence_length
            # output: batch_size * sentence_length * vec_dim
            if name == 'alpha':
                return tf.matmul(softmax, output)
            elif name == 'beta':
                return tf.matmul(tf.transpose(softmax, [0, 2, 1]), output)

if __name__ == '__main__':
    tf.set_random_seed(1)
    model = ESIM()
    model.train('dev')
