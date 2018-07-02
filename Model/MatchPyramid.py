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

class MatchPyramid():

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()

        self.lr = 0.0005
        self.keep_prob = 0.5
        self.l2_reg = 0.004
        self.sentence_length = self.preprocessor.max_length

        self.vec_dim = self.embedding.vec_dim
        self.hidden_dim = 20

        self.num_classes = 2
        self.batch_size = 128
        self.n_epoch = 50

        self.cosine = True
        self.psize1 = 2
        self.psize2 = 4

    def define_model(self):
        self.left_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='left_sentence')
        self.right_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='right_sentence')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        self.left_length = tf.placeholder(tf.int32, shape=[None])
        self.right_length = tf.placeholder(tf.int32, shape=[None])
        #self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')
        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index',shape=(None, self.sentence_length, self.sentence_length, 3))

        # Input Encoding
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

        left_seq_length, left_mask = tool.length(self.left_sentence)
        right_seq_length, right_mask = tool.length(self.right_sentence)

        # Indecator Function
        with tf.name_scope('Indecator_Function'):
            # Dot Product
            M = tf.einsum('abd,acd->abc', left_inputs, right_inputs)
            # Cosine
            if self.cosine == True:
                norm1 = tf.sqrt(tf.reduce_sum(tf.square(left_inputs), axis=2, keepdims=True))
                norm2 = tf.sqrt(tf.reduce_sum(tf.square(right_inputs), axis=2, keepdims=True))
                M = M / tf.einsum('abd,acd->abc', norm1, norm2)
            M = tf.expand_dims(M, 3)

        with tf.name_scope('convolution_1'):
            conv1 = tf.layers.conv2d(M, filters=8, kernel_size=[2, 4],
                                     strides=1, padding='SAME',
                                     activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32),
                                     name='conv1')


        with tf.name_scope('dynamic_pooling_1'):
            '''
            self.dpool_index: batchsize * length * length * 3
            3: 
                0:index
                1:选择的row
                2:选择的col
            conv1_expand:
            '''
            conv1_expand = tf.gather_nd(conv1, self.dpool_index, name = 'conv1_expand')
            pool1 = tf.nn.max_pool(conv1_expand,
                                   ksize = [1, self.sentence_length//self.psize1, self.sentence_length//self.psize2, 1],
                                   strides = [1, self.sentence_length//self.psize1, self.sentence_length/self.psize2, 1],
                                   padding='VALID',
                                   name = 'pool1')
            pool_flatten = tf.reshape(pool1, [-1, self.psize1 * self.psize2 * 8], name='pool1_flatten')
            pool_bn = tf.layers.batch_normalization(pool_flatten)

        with tf.name_scope('MLP'):
            fc = tf.layers.dense(
                inputs = pool_bn,
                units = self.hidden_dim,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.l2_reg),
            )
            fc_bn = tf.layers.batch_normalization(fc)
            fc_dropout = tf.nn.dropout(fc_bn, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('output'):
            self.output = tf.layers.dense(
                inputs = fc_dropout,
                units = self.num_classes,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = self.l2_reg),
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


    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        '''
        get self.dpool_index
        '''
        def dpool_index(idx, len1_one, len2_one, max_len1, max_len2):
            stride_1 = 1.0 * max_len1 / len1_one
            stride_2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride_1) for i in range(max_len1)]
            idx2_one = [int(i/stride_2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape)*idx, mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        for i in range(len(len1)):
            index.append(dpool_index(i, len1[i], len2[i], max_len1, max_len2))
        return  np.array(index)

    def train(self, tag='dev'):
        save_path = config.save_prefix_path + 'MatchPyramid' + '/'

        self.define_model()

        (train_left, train_right, train_labels) = self.preprocessor.get_es_index_padding('train')
        (train_left_length, train_right_length) = self.preprocessor.get_length('train')
        length = len(train_left)
        (dev_left, dev_right, dev_labels) = self.preprocessor.get_es_index_padding('dev')
        (dev_left_length, dev_right_length) = self.preprocessor.get_length('dev')

        if tag == 'train':
            train_left.extend(dev_left)
            train_right.extend(dev_right)
            train_labels.extend(dev_labels)
            train_left_length.extend(dev_left_length)
            train_right_length.extend(dev_left_length)
            del dev_left, dev_right, dev_labels
            import gc
            gc.collect()

        global_steps = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr, name='optimizer').minimize(self.cost,global_step=global_steps)


        with tf.Session() as sess:
            # debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())

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
                    train_feed_dict = self.gen_train_dict(iteration, train_left, train_right, train_left_length, train_right_length, train_labels, True)
                    train_loss, train_acc, current_step, _ = sess.run([self.cost_non_reg, self.accuracy, global_steps, self.train_op], feed_dict = train_feed_dict)
                    if current_step % 64 == 0:
                        dev_loss = 0
                        dev_acc = 0
                        if tag == 'dev':
                            for iter in range(len(dev_labels)//self.batch_size + 1):
                                dev_feed_dict = self.gen_train_dict(iter, dev_left, dev_right, dev_left_length, dev_right_length, dev_labels, False)
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
            self.left_sentence: question_batch,
            self.right_sentence: answer_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: 1.0,
        }
        return feed_dict

    def gen_train_dict(self, iteration, train_questions, train_answers, train_left_length, train_right_length, train_labels, trainable):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]
        label_batch = train_labels[start:end]
        left_length_batch = train_left_length[start:end]
        right_length_batch = train_right_length[start:end]

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
            self.dpool_index: self.dynamic_pooling_index(left_length_batch, right_length_batch, self.sentence_length, self.sentence_length)
        }
        return feed_dict



if __name__ == '__main__':
    model = MatchPyramid()
    model.train('dev')