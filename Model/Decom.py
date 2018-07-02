import tensorflow as tf
from tqdm import tqdm
import os

import sys
sys.path.append('../')

from Config import config
from Config import tool
from Preprocessing import Preprocess
from Model.Embeddings import Embeddings

class Decomposable_Attention_Model():
    '''
    sim: 0.35
    concat-mlp: 0.37
    '''
    def __init__(self, clip_gradients=False):
        self.preprocessor = Preprocess.Preprocess()
        self.embedding = Embeddings()

        self.lr = 3e-5
        self.keep_prob = 0.5
        self.atten_keep_prob = 0.8
        self.l2_reg = 0.004
        self.atten_l2_reg = 0.0
        self.sentence_length = self.preprocessor.max_length

        self.vec_dim = self.embedding.vec_dim
        self.hidden_dim = 300

        self.num_classes = 2
        self.batch_size = 64
        self.n_epoch = 50
        self.eclipse = 1e-10
        self.max_grad_norm = 5.

        self.clip_gradients = clip_gradients

    def define_model(self):
        self.left_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='left_sentence')
        self.right_sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_length], name='right_sentence')
        self.label = tf.placeholder(tf.int32, shape=[None], name='label')
        #self.features = tf.placeholder(tf.float32, shape=[None, self.num_features], name="features")
        self.trainable = tf.placeholder(bool, shape=[], name = 'trainable')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=[], name='dropout_keep_prob')

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


    def train(self, tag='dev'):
        save_path = config.save_prefix_path + 'Decom' + '/'

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
            dropout_keep_prob = self.keep_prob
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
    tf.set_random_seed(1)
    model = Decomposable_Attention_Model()
    # model.define_model()
    model.train('dev')
