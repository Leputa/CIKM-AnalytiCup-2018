import tensorflow as tf
from tqdm import tqdm
import os

import sys
sys.path.append('../')

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Model.Embeddings import Embeddings
from tensorflow.python import debug as tf_debug

class ESIM():

    def __init__(self):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()

        self.lr = 0.001
        self.keep_rate = 0.5
        self.l2_reg = 0.004
        self.sentence_length = self.preprocessor.max_length

        self.vec_dim = self.embedding.vec_dim
        self.hidden_dim = 150

        self.num_classes = 2
        self.batch_size = 128
        self.n_epoch = 20
        self.eclipse = 1e-10

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

            left__inputs = tf.cond(self.trainable,
                                      lambda: tf.nn.embedding_lookup(embedding_matrix, self.left_sentence),
                                      lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.left_sentence))
            right_inputs = tf.cond(self.trainable,
                                    lambda: tf.nn.embedding_lookup(embedding_matrix, self.right_sentence),
                                    lambda: tf.nn.embedding_lookup(embedding_matrix_fixed, self.right_sentence))

            left_inputs = tf.nn.dropout(left__inputs, self.dropout_keep_prob)
            right_inputs = tf.nn.dropout(right_inputs, self.dropout_keep_prob)

        # get lengths of unpadded sentences
        # x_seq_length: batch_size * 1
        # x_mask: batch_size * max_length * 1
        left_seq_length, left_mask = tool.length(self.left_sentence)
        right_seq_length, right_mask = tool.length(self.right_sentence)

        # Input Encoding
        with tf.name_scope("BiLSTM"):
            # x_outputs_bi: <batchsize * max_length * hidden_dim>, <batchsize * max_length * hiddden_dim>
            left_outputs, _ = tool.biLSTM(left_inputs, dim=self.hidden_dim, seq_len=left_seq_length, name='left_lstm_cell')
            right_outputs, _ = tool.biLSTM(right_inputs, dim=self.hidden_dim, seq_len=right_seq_length, name='right_lstm_cell')
            # x_outputs: bathsize * max_length * (2*hidden_dim)
            left_outputs = tf.concat(left_outputs, axis=2)
            right_outputs = tf.concat(right_outputs, axis=2)

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
            v_left, _ = tool.biLSTM(m_a, dim=self.hidden_dim, seq_len=left_seq_length, name='v_left')
            v_right, _ = tool.biLSTM(m_b, dim=self.hidden_dim, seq_len=right_seq_length, name='v_right')

            v_left_output = tf.concat(v_left, axis=2)
            v_right_output = tf.concat(v_right, axis=2)

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
            self.W_mlp = tf.get_variable(shape=(self.hidden_dim*8, self.hidden_dim), initializer=xavier_init, name='W_mlp')
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
            alpha_list = []
            for i in range(self.sentence_length):
                if name == 'alpha':
                    softmax_ = tf.expand_dims(softmax[:,i,:], -1)
                elif name == 'beta':
                    softmax_ = tf.expand_dims(softmax[:,:,i], -1)

                alpha_list.append(tf.reduce_sum(
                    tf.multiply(output, softmax_),
                    axis=1,
                    keep_dims=True
                ))

            return tf.concat(alpha_list, axis=1)

    def train(self, tag='dev'):
        save_path = config.save_prefix_path + 'ESIM' + '/'

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
        self.train_op = tf.train.AdamOptimizer(self.lr, name='optimizer').minimize(self.cost, global_step=global_steps)


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

    def gen_train_dict(self, iteration, train_questions, train_answers, train_labels, trainable):
        start = iteration * self.batch_size
        end = min((start + self.batch_size), len(train_questions))
        question_batch = train_questions[start:end]
        answer_batch = train_answers[start:end]
        label_batch = train_labels[start:end]

        if trainable == True:
            dropout_keep_prob = self.keep_rate
        else:
            dropout_keep_prob = 1.0

        feed_dict = {
            self.left_sentence: question_batch,
            self.right_sentence: answer_batch,
            self.label: label_batch,
            self.trainable: trainable,
            self.dropout_keep_prob: dropout_keep_prob,
        }
        return feed_dict

if __name__ == '__main__':
    tf.set_random_seed(2018)
    model = ESIM()
    model.train('dev')
