import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")

from Config import config
from Config import tool
from Model.BaseDeepModel import BaseDeepModel


class AB_CNN(BaseDeepModel):

    def __init__(self, model_type="ABCNN3", lang = 'es'):
        super().__init__(lang)
        self.model_type = model_type

        if self.lang == 'es':
            self.lr = 0.002
            self.batch_size = 64
            self.n_epoch = 20

            self.w = 4
            self.l2_reg = 0.001
            self.di = 32                              # The number of convolution kernels
            self.hidden_dim = 256
            self.keep_prob = 0.5

            self.num_layers = 2

        elif self.lang == 'en':
            self.lr = 0.002
            self.batch_size = 64
            self.n_epoch = 4

            self.w = 4
            self.l2_reg = 0.001
            self.di = 32  # The number of convolution kernels
            self.hidden_dim = 8
            self.keep_prob = 0.5

            self.num_layers = 2

        self.num_features = self.get_feature_num(model_type)

        # self.vocab_size = 6119

    def define_model(self):
        self.left_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question')
        self.right_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix = self.embedding.get_embedding_matrix(self.lang)
            embedding_matrix = tf.Variable(embedding_matrix, trainable=True, name='embedding')
            # embedding_matrix = tf.get_variable('embedding', [self.vocab_size, self.vec_dim], dtype=tf.float32)
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            question_inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.left_sentence),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.left_sentence))
            answer_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.right_sentence),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.right_sentence))

            question_inputs = tf.transpose(question_inputs, perm=[0, 2, 1])
            answer_inputs = tf.transpose(answer_inputs, perm=[0, 2, 1])

            question_expanded = tf.expand_dims(question_inputs, -1)
            answer_expanded = tf.expand_dims(answer_inputs, -1)

        with tf.name_scope('all_pooling'):
            question_ap_0 = self.all_pool(variable_scope='input-question', x=question_expanded)
            answer_ap_0 = self.all_pool(variable_scope='input-answer', x=answer_expanded)

        question_wp_1, question_ap_1, answer_wp_1, answer_ap_1 = self.CNN_layer(variable_scope='CNN-1', x1=question_expanded, x2=answer_expanded, d=self.vec_dim)
        sims = [self.cos_sim(question_ap_0, answer_ap_0), self.cos_sim(question_ap_1, answer_ap_1)]
        # sims = [self.get_sim(question_ap_0, answer_ap_0, self.vec_dim, 'ap_0'), self.get_sim(question_ap_1, answer_ap_1, self.di, 'ap_1')]

        if self.num_layers > 1:
            _, question_ap_2, _, answer_ap_2 = self.CNN_layer(variable_scope="CNN-2", x1=question_wp_1, x2=answer_wp_1, d=self.di)
            self.question_test = question_ap_2
            self.answer_test = answer_ap_2
            sims.append(self.cos_sim(question_ap_2, answer_ap_2))
            # sims.append(self.get_sim(question_ap_2, answer_ap_2, self.di, 'ap_2'))

        with tf.variable_scope('output_layer'):
            self.output_features = tf.stack(sims, axis=1, name='output_features')
            self.output_features = tf.concat([self.output_features, self.features], axis=1)


            self.output_features = tf.layers.batch_normalization(self.output_features)
            self.output_features = tf.nn.dropout(self.output_features, self.dropout_keep_prob, name='hidden_output_drop')

            self.fc = tf.contrib.layers.fully_connected(
                inputs = self.output_features,
                num_outputs= self.hidden_dim,
                activation_fn = tf.nn.tanh,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                biases_initializer=tf.constant_initializer(1e-02),
                scope="FC"
            )
            self.fc_bn = tf.layers.batch_normalization(self.fc)
            self.hidden_drop = tf.nn.dropout(self.fc, self.dropout_keep_prob, name='fc_drop')

            self.estimation = tf.contrib.layers.fully_connected(
                inputs = self.hidden_drop,
                num_outputs= self.num_classes,
                activation_fn = None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC_2"
            )

        self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]

        self.cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.label)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        )

        self.cost_non_reg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.label))

        with tf.name_scope('acc'):
            correct = tf.nn.in_top_k(self.estimation, self.label, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def CNN_layer(self, variable_scope, x1, x2, d):
        with tf.variable_scope(variable_scope):
            # 在输入层加入注意力
            if self.model_type == 'ABCNN1' or self.model_type == 'ABCNN3':
                with tf.name_scope('att_mat'):
                    # [sentence_length, d]
                    # question和answer共享同一个矩阵aW
                    aW = tf.get_variable(name = 'aW',
                                         shape = (self.sentence_length, d),
                                         initializer = tf.contrib.layers.xavier_initializer(),
                                         regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg))
                    # [batch_size, sentence_length, sentence_length]
                    att_mat_A = self.make_attention_mat(x1, x2)

                    # tf.einsum("ijk,kl->ijl", att_mat_A, aW) [batch_size, sentence_length, d]
                    # tf.matrix_transpose(_____)  [batch_size, d, sentence_length]
                    # tf.expand_dims(_____)  [batch_size, d, sentence_length, 1]
                    x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat_A, aW)), -1)
                    x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat_A), aW)), -1)

                    # [batch_size, d, sentence_length, 2]
                    x1 = tf.concat([x1, x1_a], axis=-1)
                    x2 = tf.concat([x2, x2_a], axis=-1)

            # 这个reuse很迷
            question_conv = self.convolution(x=self.pad_for_wide_conv(x1), d=d, reuse=False, name_scope='question')
            answer_conv = self.convolution(x=self.pad_for_wide_conv(x2), d=d, reuse=True, name_scope='answer')

            question_attention, answer_attention = None, None

            if self.model_type == 'ABCNN2' or self.model_type == 'ABCNN3':
                # matrix A [batch_size, sentence_length + w - 1, sentence_length + w - 1]
                att_mat_A = self.make_attention_mat(question_conv, answer_conv)
                # [batch_size, sentence_length + w - 1]
                question_attention, answer_attention = tf.reduce_sum(att_mat_A, axis=2), tf.reduce_sum(att_mat_A, axis=1)

            question_wp = self.w_pool(variable_scope='question', x=question_conv, attention=question_attention)
            question_ap = self.all_pool(variable_scope='question', x=question_conv)
            answer_wp = self.w_pool(variable_scope='answer', x=answer_conv, attention=answer_attention)
            answer_ap = self.all_pool(variable_scope='answer', x=answer_conv)

            return question_wp,question_ap,answer_wp,answer_ap

    def get_sim(self, x1, x2, d, name_scope):
        with tf.variable_scope("similarity_" + name_scope):
            M = tf.get_variable(
                name = 'M',
                shape = [d, d],
                initializer = tf.contrib.layers.xavier_initializer()
            )

            x1_trans = tf.matmul(x1, M)
            sims = tf.reduce_sum(tf.multiply(x1_trans, x2), axis=1)
            return sims


    def w_pool(self, variable_scope, x, attention):
        '''
        :param viriable_scope:
        :param x: [batch_size, di, sentence_length + w - 1, 1 or 2]
        :param attention: [batch_size, sentence_length + w -1]
        :return:
        '''
        with tf.variable_scope(variable_scope + '-w_pool'):
            if self.model_type == 'ABCNN2' or self.model_type == 'ABCNN3':
                pools = []
                # [batch, s+w-1] => [batch, s+w-1, 1, 1] => [batch, 1, s+w-1, 1]
                attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])
                for i in range(self.sentence_length):
                    pools.append(tf.reduce_sum(
                        x[:, :, i: i+self.w, :] * attention[:, :, i: i+self.w, :],
                        axis=2,
                        keepdims=True
                    ))
                w_ap = tf.concat(pools, axis=2, name='w_ap')
            else:
                w_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    pool_size=(1, self.w),   #用w作为卷积窗口可以还原句子长度
                    strides=1,
                    padding='VALID',
                    name='w_ap'
                )
            # w_ap [batch_size, di, sentence_length, 1 or 2]
            return w_ap


    def all_pool(self, variable_scope, x):
        # al-op
        with tf.variable_scope(variable_scope + '-all_pool'):
            if variable_scope.startswith('input'):
                # 对输入层做all-pooling
                pool_width = self.sentence_length
                d = self.vec_dim
            else:
                # 对最后的巻积层做all-pooling
                pool_width = self.sentence_length + self.w - 1
                d = self.di

            all_ap = tf.layers.average_pooling2d(
                inputs = x,
                pool_size = (1, pool_width),
                strides = 1,
                padding= 'VALID',
                name = 'all_ap'
            )
            # [batch_size, di, 1, 1]

            # [batch_size, di]
            all_ap_reshaped = tf.reshape(all_ap, [-1, d])
            return all_ap_reshaped


    def make_attention_mat(self, x1, x2):
        # x1  [batch_size, vec_dim, sentence_length, 1]
        # tf.matrix_transpose(x2) [batch_size, vec_dim, 1, sentence_length]

        # 广播产生一个 [sentence_length_0, sentence_length_1]的矩阵
        # x1 - tf.matrix_transpose(x2)  [batch_size, vec_dim, sentence_length, sentence_length]
        # euclidean [bath_size, sentence_length, sentence_length]
        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1) + self.eclipse)
        return 1 / (1 + euclidean)

    def pad_for_wide_conv(self, x):
        # 左右各填充self.w - 1个0

        # 填充前 [batch_size, d, sentence_length, 1] or [batch_size, d, sentence_length, 2]
        # 填充后 [batch_size, d, sentence_length - 2*(w-1), 1] or [batch_size, d, sentence_length - 2*(w-1), 2]
        return tf.pad(x, np.array([[0, 0], [0, 0], [self.w - 1, self.w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")


    def convolution(self, x, d, reuse, name_scope):
        # 滑窗卷积
        with tf.name_scope(name_scope + '-conv'):
            with tf.variable_scope("conv") as scope:
                conv = tf.contrib.layers.conv2d(
                    inputs = x,
                    num_outputs = self.di,
                    kernel_size = (d, self.w),
                    stride = 1,
                    padding = 'VALID',
                    activation_fn = tf.nn.tanh,
                    weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    weights_regularizer = tf.contrib.layers.l2_regularizer(scale = self.l2_reg),
                    biases_initializer = tf.constant_initializer(1e-4),
                    reuse = reuse,
                    trainable = True,
                    scope = scope,
                )
            # output [batch_size, 1, sentence_length + w - 1, di]

            # conv_trans: [batch_size, di, sentence_length + w - 1, 1]
            conv_trans = tf.transpose(conv, [0, 3, 2, 1], name = 'conv_trans')
            return conv_trans

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1*v2, axis=1, name='cos_sim')

        return dot_products / (norm1 * norm2)


if __name__ == '__main__':
    tf.set_random_seed(1)
    ABCNN = AB_CNN(model_type='ABCNN3', lang='es')
    # ABCNN.train('dev', ABCNN.model_type)
    # ABCNN.train('train', ABCNN.model_type)
    # ABCNN.test(ABCNN.model_type)
    ABCNN.cv(ABCNN.model_type, 4)


