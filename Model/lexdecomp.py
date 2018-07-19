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

        self.lr = 0.001
        self.batch_size = 64
        self.n_epoch = 3

        self.num_features = 3
        self.sentence_length = self.preprocessor.max_length
        self.w = 4
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 32
        self.vec_dim = self.embedding.vec_dim
        self.hidden_dim = 256

        self.eclipse = 1e-10
        self.num_classes = 2
        self.keep_prob = 0.5
        self.l2_reg = 0.001


    def define_model(self):
        self.left_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question')
        self.right_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
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

        with tf.name_scope('sematic_matching'):
            A = self.make_word_sim(left_inputs, right_inputs)
            # A = A * self.make_attention_mat(left_inputs, right_inputs)
            left_hat = self.semantic_match(right_inputs, A, self.w, scope='left', type='local_w')
            right_hat = self.semantic_match(left_inputs, tf.transpose(A, [0, 2, 1]), self.w, scope='right', type='local_w')

        with tf.name_scope('Decomposition'):
            left_pos, left_neg = self.decompose(left_inputs, left_hat, scope='left', type='linear')
            right_pos, right_neg = self.decompose(right_inputs, right_hat, scope='right', type='linear')
            left_decomp = tf.expand_dims(tf.concat([left_pos, left_neg], axis=-1), -1)
            right_decomp = tf.expand_dims(tf.concat([right_pos, right_neg], axis=-1), -1)


        with tf.name_scope('Composition'):
            sims = []
            pooled_left = []
            pooled_right = []
            for i, filter_size in enumerate(self.filter_sizes):
                left_conv = self.convolution(left_decomp, 2 * self.vec_dim, filter_size, self.num_filters, 'conv_left')
                left_pooled = self.max_pool(left_conv, filter_size, 'max_pooling_left')
                left_pooled_flatten = tf.reshape(left_pooled, [-1, self.num_filters], name='left_pooled_flatten')

                right_conv = self.convolution(right_decomp, 2 * self.vec_dim, filter_size, self.num_filters, 'conv_right')
                right_pooled = self.max_pool(right_conv, filter_size, 'max_pooling_right')
                right_pooled_flatten = tf.reshape(right_pooled, [-1, self.num_filters], name='right_pooled_flatten')

                sims.append(self.get_sim(left_pooled_flatten, right_pooled_flatten, self.num_filters, str(filter_size)))
                #sims.append(self.get_cos_sim(left_pooled_flatten, right_pooled_flatten, str(filter_size)))
                #pooled_left.append(left_pooled_flatten)
                #pooled_right.append(right_pooled_flatten)

            sims = tf.concat(sims, axis=-1)
            #pooled_left = tf.concat(pooled_left, axis=-1)
            #pooled_right = tf.concat(pooled_right, axis=-1)

        with tf.variable_scope('output_layer'):
            #self.output_features = sims
            #self.output_features = tf.concat([pooled_left, pooled_right, sims], axis=-1)
            self.output_features = tf.concat([sims, self.features], axis=1)
            self.output_features = tf.layers.batch_normalization(self.output_features)
            self.output_features = tf.nn.dropout(self.output_features, self.dropout_keep_prob, name='hidden_output_drop')

            self.fc = tf.contrib.layers.fully_connected(
                inputs = self.output_features,
                num_outputs= self.hidden_dim,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                biases_initializer=tf.constant_initializer(1e-02),
                scope="FC"
            )
            self.fc_bn = tf.layers.batch_normalization(self.fc)
            self.fc_bn = tf.nn.relu(self.fc_bn)
            self.hidden_drop = tf.nn.dropout(self.fc_bn, self.dropout_keep_prob, name='fc_drop')

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

    def get_cos_sim(self, x1, x2, name_scope):
        with tf.variable_scope('cos_sim_' + name_scope):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
            dot_products = tf.reduce_sum(x1 * x2, axis=1, name='cos_sim')

            cos_sim = tf.expand_dims(dot_products / (norm1 * norm2), -1)
            return cos_sim


    def get_sim(self, x1, x2, d, name_scope):
        with tf.variable_scope("similarity_" + name_scope):
            M = tf.get_variable(
                name = 'M',
                shape = [d, d],
                initializer = tf.contrib.layers.xavier_initializer()
            )

            x1_trans = tf.matmul(x1, M)
            sims = tf.reduce_sum(tf.multiply(x1_trans, x2), axis=1, keepdims=True)
            return sims

    def max_pool(self, x, filter_size, name_scope):
        with tf.name_scope(name_scope + str(filter_size)):
            pool_width = self.sentence_length - filter_size + 1

            pooled = tf.layers.max_pooling2d(
                inputs=x,
                pool_size=(pool_width, 1),
                strides=1,
                padding='VALID',
            )
            # [batch_size, 1, 1, num_filters]
            return pooled

    def convolution(self, x, d, filter_size, num_filters, name_scope):
        with tf.name_scope(name_scope + str(filter_size)):
            conv = tf.contrib.layers.conv2d(
                inputs=x,
                num_outputs=num_filters,
                kernel_size=(filter_size, d),
                stride=1,
                padding='VALID',
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                biases_initializer=tf.constant_initializer(0.1),
                trainable=True
            )
            # [batch_size, |s| - filter_size + 1, 1, num_filters]
            return conv

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
                indices = tf.tile(tf.expand_dims(tf.range(start=0, limit=self.sentence_length, dtype='int64'), axis=0), [50, 1])
                mask = tf.cast(((indices>=lower)&(indices<=upper)), 'float32')
                return tf.matmul(A*mask, X)/tf.expand_dims(tf.reduce_sum(A*mask, axis=-1), axis=-1)

    def make_word_sim(self, x, y):
        dot = tf.einsum('abd,acd->abc', x, y)
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(x), axis=2, keepdims=True))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(y), axis=2, keepdims=True))
        return dot / tf.einsum('abd,acd->abc', norm1, norm2)

    def make_attention_mat(self, x1, x2):
        # x1  [batch_size, vec_dim, sentence_length, 1]
        # tf.matrix_transpose(x2) [batch_size, vec_dim, 1, sentence_length]

        # 广播产生一个 [sentence_length_0, sentence_length_1]的矩阵
        # x1 - tf.matrix_transpose(x2)  [batch_size, vec_dim, sentence_length, sentence_length]
        # euclidean [bath_size, sentence_length, sentence_length]
        x1 = tf.expand_dims(tf.transpose(x1, [0, 2, 1]), -1)
        x2 = tf.expand_dims(tf.transpose(x2, [0, 2, 1]), -1)

        euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1) + self.eclipse)
        return 1 / (1 + euclidean)

    def train(self, tag='dev'):
        save_path = config.save_prefix_path + 'LexDecomp' + '/'

        self.define_model()

        (train_left, train_right, train_labels) = self.preprocessor.get_es_index_padding('train')
        (train_left_swap, train_right_swap, train_labels_swap) = self.preprocessor.swap_data('train', 'padding')
        train_left.extend(train_left_swap)
        train_right.extend(train_right_swap)
        train_labels.extend(train_labels_swap)

        train_features = self.Feature.addtional_feature('train')
        train_features = np.vstack([train_features, train_features])

        (dev_left, dev_right, dev_labels) = self.preprocessor.get_es_index_padding('dev')
        dev_features = self.Feature.addtional_feature('dev')

        if tag == 'train':
            (dev_left_swap, dev_right_swap, dev_labels_swap) = self.preprocessor.swap_data('dev', 'padding')
            dev_left.extend(dev_left_swap)
            dev_right.extend(dev_right_swap)
            dev_labels.extend(dev_labels_swap)
            dev_features = np.vstack([dev_features, dev_features])

            train_left.extend(dev_left)
            train_right.extend(dev_right)
            train_labels.extend(dev_labels)
            train_features = np.vstack([train_features, dev_features])

            del dev_left, dev_right, dev_labels, dev_features
            import gc
            gc.collect()

        length = len(train_left)
        shuffle_index = np.random.permutation(length)
        train_left = np.array(train_left)[shuffle_index]
        train_right = np.array(train_right)[shuffle_index]
        train_labels = np.array(train_labels)[shuffle_index]
        train_features = train_features[shuffle_index]

        global_steps = tf.Variable(0, name='global_step', trainable=False)

        self.train_op = tf.train.AdamOptimizer(self.lr, name='optimizer').minimize(self.cost,global_step=global_steps)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

            # debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            if os.path.exists(save_path):
                try:
                    ckpt = tf.train.get_checkpoint_state(save_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                except:
                    ckpt = None
            else:
                os.makedirs(save_path)

            if length % self.batch_size == 0:
                iters = length // self.batch_size
            else:
                iters = length // self.batch_size + 1

            for epoch in range(self.n_epoch):
                for iteration in range(iters):
                    train_feed_dict = self.gen_train_dict(iteration, train_left, train_right, train_features, train_labels, True)
                    train_loss, train_acc, current_step, _ = sess.run([self.cost_non_reg, self.accuracy, global_steps, self.train_op], feed_dict = train_feed_dict)
                    if current_step % 64 == 0:
                        dev_loss = 0
                        dev_acc = 0
                        if tag == 'dev':
                            for iter in range(len(dev_labels)//self.batch_size + 1):
                                dev_feed_dict = self.gen_train_dict(iter, dev_left, dev_right, dev_features, dev_labels, False)
                                dev_loss += self.cost_non_reg.eval(feed_dict = dev_feed_dict)
                                dev_acc += self.accuracy.eval(feed_dict = dev_feed_dict)
                            dev_loss = dev_loss/(len(dev_labels)//self.batch_size + 1)
                            dev_acc = dev_acc/(len(dev_labels)//self.batch_size + 1)
                        print("**********************************************************************************************************")
                        print("Epoch {}, Iteration {}, train loss: {:.4f}, train accuracy: {:.4f}%.".format(epoch,
                                                                                                            current_step,
                                                                                                            train_loss,
                                                                                                            train_acc * 100))
                        if tag == 'dev':
                            print("Epoch {}, Iteration {}, val loss: {:.4f}, val accuracy: {:.4f}%.".format(epoch,
                                                                                                              current_step,
                                                                                                              dev_loss,
                                                                                                              dev_acc * 100))
                            print("**********************************************************************************************************")
                        checkpoint_path = os.path.join(save_path, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step = current_step)

    def test(self):
        save_path = config.save_prefix_path + 'LexDecomp' + '/'
        assert os.path.isdir(save_path)

        test_left, test_right = self.preprocessor.get_es_index_padding('test')
        test_features = self.Feature.addtional_feature('test')

        tf.reset_default_graph()
        self.define_model()
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            test_results = []
            init.run()
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, os.path.join(save_path, 'best_model.ckpt'))

            for step in tqdm(range(len(test_left)//self.batch_size + 1)):
                test_feed_dict = self.gen_test_dict(step, test_left, test_right, test_features, False)
                pred = sess.run(self.prediction, feed_dict = test_feed_dict)
                test_results.extend(pred.tolist())

        with open(config.output_prefix_path + 'LexDecomp' + '-submit.txt', 'w') as fr:
            for result in test_results:
                fr.write(str(result) + '\n')

    def gen_test_dict(self, iteration, train_questions, train_answers, features, trainable = False):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]
        features_batch = features[start:end]

        feed_dict = {
            self.left_sentence: question_batch,
            self.right_sentence: answer_batch,
            self.features: features_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: 1.0,
        }
        return feed_dict

    def gen_train_dict(self, iteration, train_questions, train_answers, train_features, train_labels, trainable):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]
        label_batch = train_labels[start:end]
        feature_batch = train_features[start:end]

        if trainable == True:
            dropout_keep_prob = self.keep_prob
        else:
            dropout_keep_prob = 1.0

        feed_dict = {
            self.left_sentence: question_batch,
            self.right_sentence: answer_batch,
            self.label: label_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: dropout_keep_prob,
            self.features: feature_batch
        }
        return feed_dict


if __name__ == '__main__':
    tf.set_random_seed(1024)
    model = LexDecomp()
    # model.define_model()
    model.train('train')
    model.test()


