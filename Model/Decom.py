import tensorflow as tf

import sys
sys.path.append('../')

from Config import config
from Config import tool
from Model.BaseDeepModel import BaseDeepModel


class Decomposable_Attention_Model(BaseDeepModel):
    '''
    sim: 0.35
    concat-mlp: 0.37
    '''
    def __init__(self, lang = 'es',clip_gradients=False):
        super().__init__(lang)
        if lang == 'es':
            self.lr = 3e-5
            self.keep_prob = 0.5
            self.atten_keep_prob = 0.8
            self.l2_reg = 0.004
            self.atten_l2_reg = 0.0

            self.hidden_dim = 300

            self.batch_size = 64
            self.n_epoch = 50
        elif lang == 'en':
            self.lr = 3e-5
            self.keep_prob = 0.5
            self.atten_keep_prob = 0.8
            self.l2_reg = 0.004
            self.atten_l2_reg = 0.0

            self.hidden_dim = 300

            self.batch_size = 64
            self.n_epoch = 50


    def define_model(self):
        self.left_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='left_sentence')
        self.right_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='right_sentence')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        #self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

        # Input Encoding
        with tf.name_scope('embedding'):
            embedding_matrix = self.embedding.get_embedding_matrix(self.lang)
            embedding_matrix = tf.Variable(embedding_matrix, trainable=True, name='embedding')
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            left_inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.left_sentence),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.left_sentence))
            right_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.right_sentence),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.right_sentence))

        left_seq_length, left_mask = tool.length(self.left_sentence)
        right_seq_length, right_mask = tool.length(self.right_sentence)

        with tf.name_scope('Bi-LSTM'):
            left_outputs, _ = tool.biLSTM(left_inputs, self.hidden_dim, left_seq_length, 'left')
            right_outputs, _ = tool.biLSTM(right_inputs, self.hidden_dim, right_seq_length, 'right')
            left_outputs = tf.concat(left_outputs, axis=2)
            right_outputs = tf.concat(right_outputs, axis=2)

            # left_outputs_mask = left_outputs * tf.expand_dims(left_mask, -1)
            # right_outputs_mask = right_outputs * tf.expand_dims(right_mask, -1)
            #
            # left_max = tf.reduce_max(left_outputs_mask, axis=1)
            # right_max = tf.reduce_max(right_outputs_mask, axis=1)
            #
            # left_mean = tf.reduce_sum(left_outputs_mask, axis=1) / tf.expand_dims(tf.cast(left_seq_length, tf.float32), -1)
            # right_mean = tf.reduce_sum(right_outputs_mask, axis=1) / tf.expand_dims(tf.cast(right_seq_length, tf.float32), -1)

        # Attend
        with tf.name_scope('Attention'):
            hidden_dim = left_outputs.shape[-1]*2

            att_left = self.MLP(left_outputs, self.trainable, hidden_dim, 'left_Attention')
            att_right = self.MLP(right_outputs, self.trainable, hidden_dim, 'right_Attention')

            scores = tf.matmul(att_left, att_right, adjoint_b=True)
            scores_mask = tf.einsum('bi,bj->bij', left_mask, right_mask)

            right_weights = self.softmax_mask(scores, 2, scores_mask, 'right')
            left_weights = self.softmax_mask(scores, 1, scores_mask, 'left')
            left_weights = tf.transpose(left_weights, [0, 2, 1])


            alpha = tf.matmul(left_weights, left_outputs)
            beta = tf.matmul(right_weights, right_outputs)

            # alpha_max = tf.reduce_max(alpha, axis=1)
            # beta_max = tf.reduce_max(beta, axis=1)
            # alpha_mean = tf.reduce_mean(alpha, axis=1)
            # beta_mean = tf.reduce_mean(beta, axis=1)
            #
            # sims = [self.cos_sim(alpha_max, beta_max), self.cos_sim(alpha_mean, beta_mean), self.cos_sim(left_max, right_max), self.cos_sim(left_mean, right_mean)]

        # Compare
        with tf.name_scope('Compare'):
            v_input_left = tf.concat([left_inputs, beta], axis=2)
            v_input_right = tf.concat([right_inputs, alpha], axis=2)

            hidden_dim = ((alpha.shape[-1] + left_inputs.shape[-1]) // 2) * 2
            v_left = self.MLP(v_input_left, self.trainable, hidden_dim, 'left_Compare')
            v_right = self.MLP(v_input_right, self.trainable, hidden_dim, 'right_Compare')

        # Aggregate
        with tf.name_scope('Aggregate'):
            v_left_mask = v_left * tf.expand_dims(left_mask, -1)
            v_right_mask = v_right * tf.expand_dims(right_mask, -1)

            v_left_output = tf.reduce_sum(v_left_mask, axis=1)
            v_right_output = tf.reduce_sum(v_right_mask, axis=1)

            v = tf.concat([v_left_output, v_right_output], axis=1)
            v_output = self.MLP(v, self.trainable, self.hidden_dim, 'Aggregate')
            # v_left_max = tf.reduce_max(v_left_mask, axis=1)
            # v_right_max = tf.reduce_max(v_right_mask, axis=1)
            # v_left_mean = tf.reduce_sum(v_left_mask, axis=1)/tf.expand_dims(tf.cast(left_seq_length, tf.float32), -1)
            # v_right_mean = tf.reduce_sum(v_right_mask, axis=1)/tf.expand_dims(tf.cast(right_seq_length, tf.float32), -1)
            #
            # sims.append(self.cos_sim(v_left_max, v_right_max))
            # sims.append(self.cos_sim(v_left_mean, v_right_mean))
            #
            # v_output = tf.stack(sims, axis=1)


        with tf.name_scope('MLP'):
            self.output = tf.layers.dense(
                inputs = v_output,
                units = self.num_classes,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.atten_l2_reg),
            )

            self.prediction = tf.contrib.layers.softmax(self.output)[:, 1]

        with tf.name_scope('cost'):
            self.cost = tf.add(
                tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label)),
                tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            )
            self.cost_non_reg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.label))

        with tf.name_scope('acc'):
            correct = tf.nn.in_top_k(self.output, self.label, 1)
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1*v2, axis=1, name='cos_sim')

        return dot_products / (norm1 * norm2)


    def MLP(self, input, trainable, hidden_dim, name):
        with tf.variable_scope('mlp' + name):
            F_output = tf.layers.dense(
                inputs = input,
                units = hidden_dim,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.atten_l2_reg),
                activation = tf.nn.relu,
            )
            if trainable == True:
                if name == 'Aggregate':
                    keep_drop = self.keep_prob
                else:
                    keep_drop = self.atten_keep_prob
            else:
                keep_drop = 1.0
            F_bn = tf.layers.batch_normalization(F_output)
            F_drop = tf.nn.dropout(F_bn, keep_drop)
            return F_drop


    def softmax_mask(self, scores, axis, mask, name=None):
        '''
        求解alpha和belta
        :return: batch_size * sentence_length * vec_dim
        '''
        with tf.name_scope(name + 'softmax'):
            max_axis = tf.reduce_max(scores, axis=axis, keep_dims=True)
            target_exp = tf.exp(scores - max_axis) * mask
            normalize = tf.reduce_sum(target_exp, axis=axis, keep_dims=True)
            softmax = target_exp / (normalize + self.eclipse)
            return softmax

if __name__ == '__main__':
    tf.set_random_seed(1)
    model = Decomposable_Attention_Model()
    # model.define_model()
    model.train('dev')
