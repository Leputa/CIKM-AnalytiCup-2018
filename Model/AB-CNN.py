import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.append("../")

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Model.Embeddings import Embeddings
from tensorflow.python import debug as tf_debug

class AB_CNN():

    def __init__(self, model_type="ABCNN3", clip_gradients=False):
        self.model_type = model_type

        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()
        self.lr = 0.01
        self.batch_size = 64
        self.n_epoch = 5

        self.sentence_length = self.preprocessor.max_length
        self.w = 4
        self.l2_reg = 0.01
        self.di = 64                               # The number of convolution kernels
        self.vec_dim = self.embedding.vec_dim
        self.num_classes = 2
        self.num_layers = 2
        self.num_features = None

        self.clip_gradients = clip_gradients
        self.max_grad_norm = 5.
        self.eclipse = 1e-9

        self.vocab_size = 6119

    def define_model(self):
        self.question = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='question')
        self.answer = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='answer')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

        with tf.name_scope('embedding'):
            embedding_matrix = self.embedding.get_es_embedding_matrix()
            embedding_matrix = tf.Variable(embedding_matrix, trainable=True, name='embedding')
            # embedding_matrix = tf.get_variable('embedding', [self.vocab_size, self.vec_dim], dtype=tf.float32)
            embedding_matrix_fixed = tf.stop_gradient(embedding_matrix, name='embedding_fixed')

            question_inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.question),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.question))
            answer_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.answer),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.answer))

            question_inputs = tf.transpose(question_inputs, perm=[0, 2, 1])
            answer_inputs = tf.transpose(answer_inputs, perm=[0, 2, 1])

            question_expanded = tf.expand_dims(question_inputs, -1)
            answer_expanded = tf.expand_dims(answer_inputs, -1)

        with tf.name_scope('all_pooling'):
            question_ap_0 = self.all_pool(variable_scope='input-question', x=question_expanded)
            answer_ap_0 = self.all_pool(variable_scope='input-answer', x=answer_expanded)

        question_wp_1, question_ap_1, answer_wp_1, answer_ap_1 = self.CNN_layer(variable_scope='CNN-1', x1=question_expanded, x2=answer_expanded, d=self.vec_dim)
        sims = [self.cos_sim(question_ap_0, answer_ap_0), self.cos_sim(question_ap_1, answer_ap_1)]

        if self.num_layers > 1:
            _, question_ap_2, _, answer_ap_2 = self.CNN_layer(variable_scope="CNN-2", x1=question_wp_1, x2=answer_wp_1, d=self.di)
            self.question_test = question_ap_2
            self.answer_test = answer_ap_2
            sims.append(self.cos_sim(question_ap_2, answer_ap_2))

        with tf.variable_scope('output_layer'):
            self.output_features = tf.stack(sims, axis=1, name='output_features')
            self.output_features = tf.layers.batch_normalization(self.output_features)
            self.hidden_drop = tf.nn.dropout(self.output_features, self.dropout_keep_prob, name='hidden_output_drop')

            self.estimation = tf.contrib.layers.fully_connected(
                inputs = self.hidden_drop,
                num_outputs= self.num_classes,
                activation_fn = None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
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
                    pool_size=(1,self.w),   #用w作为卷积窗口可以还原句子长度
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

    def train(self, tag='dev'):
        save_path = config.save_prefix_path + self.model_type + '/'

        self.define_model()

        (train_left, train_right, train_labels) = self.preprocessor.get_es_index_padding('train')
        length = len(train_left)
        (dev_left, dev_right, dev_labels) = self.preprocessor.get_es_index_padding('dev')

        if tag == 'train':
            train_left.extend(dev_left)
            train_right.extend(dev_right)
            train_labels.extend(dev_labels)
            del dev_left, dev_right, dev_labels
            import gc
            gc.collect()

        global_steps = tf.Variable(0, name='global_step', trainable=False)

        if self.clip_gradients == True:
            optimizer = tf.train.AdagradOptimizer(self.lr)

            grads_and_vars = optimizer.compute_gradients(self.cost)
            gradients = [output[0] for output in grads_and_vars]
            variables = [output[1] for output in grads_and_vars]

            gradients = tf.clip_by_global_norm(gradients, clip_norm=self.max_grad_norm)[0]
            self.grad_norm = tf.global_norm(gradients)

            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_steps)
        else:
            #self.train_op = tf.train.AdagradOptimizer(self.lr, name='optimizer').minimize(self.cost, global_step=global_steps)
            self.train_op = tf.train.AdamOptimizer(self.lr, name='optimizer').minimize(self.cost,global_step=global_steps)


        # 为了提前停止训练
        # best_loss_test = np.infty
        # checks_since_last_progress = 0
        # max_checks_without_progress = 40
        # best_model_params = None

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

            for epoch in range(self.n_epoch):
                for iteration in range(length//self.batch_size + 1):
                    train_feed_dict = self.gen_train_dict(iteration, train_left, train_right, train_labels, True)
                    train_loss, train_acc, current_step, _ = sess.run([self.cost_non_reg, self.accuracy, global_steps, self.train_op], feed_dict = train_feed_dict)
                    if current_step % 64 == 0:
                        dev_loss = 0
                        dev_acc = 0
                        if tag == 'dev':
                            for iter in range(len(dev_labels)//self.batch_size + 1):
                                dev_feed_dict = self.gen_train_dict(iter, dev_left, dev_right, dev_labels, False)
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

            #             if test_loss < best_loss_test:
            #                 best_loss_test = test_loss
            #                 checks_since_last_progress = 0
            #                 best_model_params = tool.get_model_params()
            #             else:
            #                 checks_since_last_progress += 1
            #
            #             if checks_since_last_progress>max_checks_without_progress:
            #                 print("Early Stopping")
            #                 break
            #     if checks_since_last_progress > max_checks_without_progress:
            #         break
            #
            # if best_model_params:
            #     tool.restore_model_params(best_model_params)
            # saver.save(sess, os.path.join(save_path, 'best_model.ckpt'))

    def test(self):
        save_path = config.save_prefix_path + self.model_type + '/'
        assert os.path.isdir(save_path)

        test_left, test_right = self.preprocessor.get_es_index_padding('test')
        # test_left, test_right, _ = self.preprocessor.get_es_index_padding('dev')

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
                test_feed_dict = self.gen_test_dict(step, test_left, test_right, False)
                pred = sess.run(self.prediction, feed_dict = test_feed_dict)
                test_results.extend(pred.tolist())

        with open(config.output_prefix_path + self.model_type + '-submit.txt', 'w') as fr:
            for result in test_results:
                fr.write(str(result) + '\n')

    def gen_test_dict(self, iteration, train_questions, train_answers, trainable = False):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]

        feed_dict = {
            self.question: question_batch,
            self.answer: answer_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: 1.0,
        }
        return feed_dict

    def gen_train_dict(self, iteration, train_questions, train_answers, train_labels, trainable):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]
        label_batch = train_labels[start:end]

        if trainable == True:
            dropout_keep_prob = 0.6
        else:
            dropout_keep_prob = 1.0

        feed_dict = {
            self.question: question_batch,
            self.answer: answer_batch,
            self.label: label_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: dropout_keep_prob,
        }
        return feed_dict

if __name__ == '__main__':
    tf.set_random_seed(1)
    ABCNN = AB_CNN(model_type='ABCNN3')
    ABCNN.train('train')
    ABCNN.test()

