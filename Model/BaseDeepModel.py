import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

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
    def __init__(self, lang):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()
        self.Feature = Feature.Feature()
        self.Powerfulwords = PowerfulWord.PowerfulWord()
        self.Graph = GraphFeature.GraphFeature()
        self.lang = lang

        if lang == 'es':
            self.sentence_length = self.preprocessor.max_es_length
        elif lang == 'en':
            self.sentence_length = self.preprocessor.max_en_length

        self.n_folds = 10
        self.num_classes = 2
        self.eclipse = 1e-10
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
        if modeltype.startswith('ABCNN') or modeltype == 'LexDecomp':
            return statistic_feature

    def prepare_data(self, tag, modeltype, lang):
        if lang == 'es':
            (_, _, train_left, train_right, train_labels) = self.preprocessor.get_index_padding('train')
            (_, _, train_left_swap, train_right_swap, train_labels_swap) = self.preprocessor.swap_data('train', 'padding')
            (_, _, dev_left, dev_right, dev_labels) = self.preprocessor.get_index_padding('dev')
        elif lang == 'en':
            (train_left, train_right, _, _, train_labels) = self.preprocessor.get_index_padding('train')
            (train_left_swap, train_right_swap, _, _, train_labels_swap) = self.preprocessor.swap_data('train', 'padding')
            (dev_left, dev_right, _, _, dev_labels) = self.preprocessor.get_index_padding('dev')


        train_left.extend(train_left_swap)
        train_right.extend(train_right_swap)
        train_labels.extend(train_labels_swap)

        train_features = self.get_feature('train', modeltype)
        train_features = np.vstack([train_features, train_features])

        dev_features = self.get_feature('dev', modeltype)

        if tag == 'train':
            if lang == 'es':
                (_, _, dev_left_swap, dev_right_swap, dev_labels_swap) = self.preprocessor.swap_data('dev', 'padding')
            elif lang == 'en':
                (dev_left_swap, dev_right_swap, _, _, dev_labels_swap) = self.preprocessor.swap_data('dev', 'padding')
            dev_left.extend(dev_left_swap)
            dev_right.extend(dev_right_swap)
            dev_labels.extend(dev_labels_swap)
            dev_features = np.vstack([dev_features, dev_features])

            train_left.extend(dev_left)
            train_right.extend(dev_right)
            train_labels.extend(dev_labels)
            train_features = np.vstack([train_features, dev_features])

        length = len(train_left)
        train_left = np.array(train_left)
        train_right = np.array(train_right)
        train_labels = np.array(train_labels)
        train_features = np.array(train_features)

        if tag == 'train':
            return (train_left, train_right, train_labels, train_features), length
        if tag == 'dev':
            return (train_left, train_right, train_labels, train_features), (dev_left, dev_right, dev_labels, dev_features), length


    def train(self, tag, model_type):
        print("starting training......")

        save_path = config.save_prefix_path + self.lang + '_' + model_type + '/'

        self.define_model()

        if tag == 'train':
            (train_left, train_right, train_labels, train_features), length = self.prepare_data(tag, model_type, self.lang)
        elif tag == 'dev':
            (train_left, train_right, train_labels, train_features), (dev_left, dev_right, dev_labels, dev_features), length = self.prepare_data(tag, model_type, self.lang)

        shuffle_index = np.random.permutation(length)
        train_left = np.array(train_left)[shuffle_index]
        train_right = np.array(train_right)[shuffle_index]
        train_labels = np.array(train_labels)[shuffle_index]
        train_features = train_features[shuffle_index]

        global_steps = tf.Variable(0, name='global_step', trainable=False)

        if self.clip_gradients == True:
            optimizer = tf.train.AdamOptimizer(self.lr)

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
        save_path = config.save_prefix_path + self.lang + '_' + modeltype + '/'
        assert os.path.isdir(save_path)

        if self.lang == 'es':
            _, _, test_left, test_right = self.preprocessor.get_index_padding('test')
        elif self.lang == 'en':
            test_left, test_right, _, _ = self.preprocessor.get_index_padding('test')
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

        with open(config.output_prefix_path + self.lang + '_' + modeltype + '-submit.txt', 'w') as fr:
            for result in test_results:
                fr.write(str(result) + '\n')


    # cv后结果反而下降了，奇怪
    def cv(self, model_type, n_epoch):
        print("starting training......")

        save_dir = config.save_prefix_path + self.lang + '_' + model_type + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        (train_left, train_right, train_labels, train_features), length = self.prepare_data('train', model_type, self.lang)
        if self.lang == 'es':
            _, _, test_left, test_right = self.preprocessor.get_index_padding('test')
        elif self.lang == 'en':
            test_left, test_right, _, _ = self.preprocessor.get_index_padding('test')
        test_features = self.get_feature('test', model_type)

        oof = np.zeros(train_left.shape[0])
        sub = np.zeros(len(test_left))

        dev_auc = 0
        dev_logloss = 0

        folds = KFold(n_splits=self.n_folds, random_state=2018, shuffle=True)

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_left, train_labels)):

            fold_train_left, fold_train_right, fold_train_labels, fold_train_features = train_left[trn_idx], train_right[trn_idx], train_labels[trn_idx], train_features[trn_idx]
            fold_dev_left, fold_dev_right, fold_dev_labels, fold_dev_features = train_left[val_idx], train_right[val_idx], train_labels[val_idx], train_features[val_idx]

            tf.reset_default_graph()
            self.define_model()
            global_steps = tf.Variable(0, name='global_step', trainable=False)

            if self.clip_gradients == True:
                optimizer = tf.train.AdamOptimizer(self.lr)

                grads_and_vars = optimizer.compute_gradients(self.cost)
                gradients = [output[0] for output in grads_and_vars]
                variables = [output[1] for output in grads_and_vars]

                gradients = tf.clip_by_global_norm(gradients, clip_norm=self.max_grad_norm)[0]
                self.grad_norm = tf.global_norm(gradients)

                self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_steps)
            else:
                self.train_op = tf.train.AdamOptimizer(self.lr, name='optimizer').minimize(self.cost,
                                                                                           global_step=global_steps)
            save_path = save_dir + str(n_fold) + '/'

            with tf.Session() as sess:
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

                if length % self.batch_size == 0:
                    iters = length // self.batch_size
                else:
                    iters = length // self.batch_size + 1

                # train
                for epoch in range(n_epoch):
                    for iteration in range(iters):
                        train_feed_dict = self.gen_train_dict(iteration, fold_train_left, fold_train_right, fold_train_features, fold_train_labels, True)
                        train_loss, train_acc, current_step, _ = sess.run([self.cost_non_reg, self.accuracy, global_steps, self.train_op], feed_dict=train_feed_dict)
                    checkpoint_path = os.path.join(save_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=current_step)

                # dev
                dev = []
                for iter in range(len(fold_dev_left)//self.batch_size + 1):
                    dev_feed_dict = self.gen_test_dict(iter, fold_dev_left, fold_dev_right, fold_dev_features, False)
                    pred = sess.run(self.prediction, feed_dict=dev_feed_dict)
                    dev.extend(pred.tolist())

                dev = np.array(dev)
                tmp_auc = roc_auc_score(fold_dev_labels, dev)
                tmp_logloss = log_loss(fold_dev_labels, dev)
                dev_auc += tmp_auc
                dev_logloss += tmp_logloss

                print('\t Fold %d : %.6f auc and %.6f logloss' % (n_fold + 1, tmp_auc, tmp_logloss))
                oof[val_idx] = dev


                test_results = []
                for step in tqdm(range(len(test_left) // self.batch_size + 1)):
                    test_feed_dict = self.gen_test_dict(step, test_left, test_right, test_features, False)
                    pred = sess.run(self.prediction, feed_dict=test_feed_dict)
                    test_results.extend(pred.tolist())
                sub += np.array(test_results)

        dev_auc /= self.n_folds
        dev_logloss /= self.n_folds

        print('Average %.6f auc and %.6f logloss' % (dev_auc, dev_logloss))

        sub /= self.n_folds
        with open(config.output_prefix_path + str(n_epoch) +  "_cv_" + self.lang + '_' + model_type + '-submit.txt', 'w') as fr:
            for result in sub:
                fr.write(str(result) + '\n')

        with open(config.output_prefix_path  + str(n_epoch) + '_' + self.lang + '_' + model_type + '-oof.txt', 'w') as fr:
            for result in oof:
                fr.write(str(result) + '\n')







