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
from Preprocessing import Feature
from Preprocessing import PowerfulWord
from Preprocessing import GraphFeature

class BaseDeepModel():
    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()
        self.Feature = Feature.Feature()
        self.Powerfulwords = PowerfulWord.PowerfulWord()
        self.Graph = GraphFeature.GraphFeature()


        self.num_classes = 2
        self.eclipse = 1e-10
        self.sentence_length = self.preprocessor.max_length
        self.vec_dim = self.embedding.vec_dim

        self.clip_gradients = False
        self.max_grad_norm = 5.

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

    def get_feature_num(self, modeltype):
        return self.get_feature('test', modeltype).shape[1]

    def get_feature(self, tag, modeltype):
        statistic_feature = self.Feature.addtional_feature(tag, modeltype)
        if modeltype.startswith('ABCNN'):
            return statistic_feature
        if modeltype == 'LexDecomp':
            powerwords_feature = self.Powerfulwords.addtional_feature(tag, modeltype)
            return np.hstack([statistic_feature, powerwords_feature])

    def prepare_data(self, tag, modeltype):
        (train_left, train_right, train_labels) = self.preprocessor.get_es_index_padding('train')
        (train_left_swap, train_right_swap, train_labels_swap) = self.preprocessor.swap_data('train', 'padding')
        train_left.extend(train_left_swap)
        train_right.extend(train_right_swap)
        train_labels.extend(train_labels_swap)

        train_features = self.get_feature('train', modeltype)
        train_features = np.vstack([train_features, train_features])

        (dev_left, dev_right, dev_labels) = self.preprocessor.get_es_index_padding('dev')
        dev_features = self.get_feature('dev', modeltype)

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

        length = len(train_left)
        shuffle_index = np.random.permutation(length)
        train_left = np.array(train_left)[shuffle_index]
        train_right = np.array(train_right)[shuffle_index]
        train_labels = np.array(train_labels)[shuffle_index]
        train_features = train_features[shuffle_index]

        if tag == 'train':
            return (train_left, train_right, train_labels, train_features), length
        if tag == 'dev':
            return (train_left, train_right, train_labels, train_features), (dev_left, dev_right, dev_labels, dev_features), length


    def train(self, tag, model_type):
        print("starting training......")

        save_path = config.save_prefix_path + model_type + '/'

        self.define_model()

        if tag == 'train':
            (train_left, train_right, train_labels, train_features), length = self.prepare_data(tag, model_type)
        elif tag == 'dev':
            (train_left, train_right, train_labels, train_features), (dev_left, dev_right, dev_labels, dev_features), length = self.prepare_data(tag, model_type)

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

    def test(self, modeltype):
        save_path = config.save_prefix_path + modeltype + '/'
        assert os.path.isdir(save_path)

        test_left, test_right = self.preprocessor.get_es_index_padding('test')
        test_features = self.get_feature('test', modeltype)

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

        with open(config.output_prefix_path + modeltype + '-submit.txt', 'w') as fr:
            for result in test_results:
                fr.write(str(result) + '\n')