import tensorflow as tf
import numpy as np
import time
import math
from six.moves import xrange
import scipy.io
import os
import collections
from sklearn.metrics import roc_curve, auc
from attention_operation import *
from metric import *
import random

class UA(object):
    dic = {}
    def __init__(self, config):
        self.task = config['task']
        self.num_features = config['num_features']
        self.steps = config['steps']

        self.num_layers = config['num_layers']
        self.hidden_units = config['hidden_units']
        self.embed_size = config['embed_size']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.save_iter = config['save_iter']
        self.max_epoch = config['max_epoch']

        self.train_x = config['train_x']
        self.train_y = config['train_y']
        self.val_x = config['val_x']
        self.val_y = config['val_y']
        self.eval_x = config['eval_x']
        self.eval_y = config['eval_y']
        self.train_range = np.array(range(len(config['train_x'])))
        self.test_range = np.array(range(len(config['eval_x'])))
        self.sess = config['sess']

        self.x = tf.placeholder(shape=[None, config['steps'], config['num_features']], dtype=tf.float32, name='data')
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels')
        self.input_keep_prob = tf.placeholder('float')
        self.output_keep_prob = tf.placeholder('float')
        self.state_keep_prob = tf.placeholder('float')
        self.num_sampling = config['num_sampling']
        self.lamb = config['lamb']

    def build_model(self):
        print 'Start building a model.'
        def single_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
	    return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                                 input_keep_prob=self.input_keep_prob, \
                                                 output_keep_prob=self.output_keep_prob,
                                                 state_keep_prob=self.state_keep_prob, \
                                                 dtype=tf.float32
                                                 )

        cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

        with tf.variable_scope('embedded'):
            self.V = tf.get_variable('v_weight', shape=[self.num_features, self.embed_size], dtype=tf.float32)
        with tf.variable_scope('output_v'):
            self.out_weight = tf.get_variable('weight', shape=[self.embed_size, 1])
            self.out_bias   = tf.get_variable('bias', shape=[1])
            self.sigma_weight = tf.get_variable('sigma_weight', shape=[self.hidden_units, 1])
            self.sigma_bias   = tf.get_variable('sigma_bias', shape=[1])

        v_emb = []
        with tf.variable_scope('embedded', reuse=True):
            for _j in range(self.steps):
                self.V = tf.get_variable(name='v_weight')
                embbed = tf.matmul(self.x[:, _j, :], self.V)
                v_emb.append(embbed)
            self.embedded_v = tf.reshape(tf.concat(v_emb, 1), [-1, self.steps, self.hidden_units])

        #Reverse embedded_v
        reversed_v_outputs = tf.reverse(self.embedded_v, [1])

        with tf.variable_scope("myrnns_alpha") as scope:
            alpha_rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                                     reversed_v_outputs,
                                                     dtype=tf.float32
                                                     )

        with tf.variable_scope("myrnns_beta") as scope:
            beta_rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                                    reversed_v_outputs,
                                                    dtype=tf.float32
                                                    )

        #alpha
        alpha_embed_output = attention_op('alpha', alpha_rnn_outputs, self.hidden_units, self.embed_size, self.steps)
        self.rev_alpha_embed_output = tf.reverse(alpha_embed_output, [1])

        #beta
        beta_embed_output = attention_op('beta', beta_rnn_outputs, self.hidden_units, self.embed_size, self.steps)
        self.rev_beta_embed_output = tf.reverse(beta_embed_output, [1])

        # attention_sum 
        c_i = tf.reduce_sum(self.rev_alpha_embed_output * (self.rev_beta_embed_output * self.embedded_v), 1)
        
        #mu
        logits = tf.matmul(c_i, self.out_weight) + self.out_bias
        self.preds = tf.nn.sigmoid(logits)

        all_variables = tf.trainable_variables()
        l2_losses = []
        for variable in all_variables:
            variable = tf.cast(variable, tf.float32)
            l2_losses.append(tf.nn.l2_loss(variable))
        regul = self.lamb*tf.reduce_sum(l2_losses)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y))
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss + regul)
        print ('Done with builing the model.')


    def data_iteration(self, data_x, data_y, is_train=True):
        data_range=None
        if is_train:
            data_range = self.train_range
            random.shuffle(data_range)
        
            batch_len = len(data_range) // self.batch_size
            for i_ in xrange(batch_len):
                b_idx = data_range[self.batch_size*i_:self.batch_size*(i_+1)]

                batch_inputs = np.zeros((self.batch_size, self.steps, self.num_features), np.float32)
                batch_labels = np.zeros((self.batch_size, 1), np.float32)

                for j_ in range(self.batch_size):
                    inp   = np.copy(data_x[b_idx[j_]])
                    label = np.copy(data_y[b_idx[j_]])
                    batch_inputs[j_] = inp
                    batch_labels[j_] = label
                yield batch_inputs, batch_labels

        else:
            yield data_x, data_y


    def run_epoch(self, ops, data_x, data_y, is_train=True):
        total_preds=[]
        total_labels=[]
        total_loss=[]
        input_keep_prob = 1.
        output_keep_prob = 1.
        state_keep_prob = 1.

        if is_train:
            input_keep_prob = 0.95
            output_keep_prob = 0.75
            state_keep_prob = 0.95

        for step, (data_in, label_in) in enumerate(self.data_iteration(data_x, data_y, is_train)):
            _, loss, preds = self.sess.run([ops, self.loss, self.preds],
                             feed_dict = {
                                          self.x: data_in,
                                          self.y: label_in,
                                          self.input_keep_prob: input_keep_prob,
                                          self.output_keep_prob: output_keep_prob,
                                          self.state_keep_prob: state_keep_prob,
                                          })
            total_preds.append(preds)
            total_labels.append(label_in)
            total_loss.append(loss)

        total_loss = np.mean(total_loss, axis=0)
        total_preds = np.concatenate(total_preds, axis=0)
        total_labels = np.concatenate(total_labels, axis=0)

        eval_preds = total_preds

        roc, auc = ROC_AUC(total_preds, total_labels)
        total_preds = total_preds >= 0.5
        acc = accuracy(total_preds, total_labels)
        return total_loss, auc, acc, total_preds if is_train else eval_preds

    
    def run(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)

        for i_ in range(self.max_epoch):
            train_loss, train_auc, train_acc, _ = self.run_epoch(self.optimize, self.train_x, self.train_y, is_train=True)
            print(" [*] Epoch: %d,      Train loss: %.4f,      Train AUC: %.4f,      Train ACC: %.4f" % (i_+1, train_loss, train_auc, train_acc))
            
            total_val_preds=[]
            total_val_loss=[]
            for sample in range(self.num_sampling):
                valid_loss, _, _, valid_preds = self.run_epoch(tf.no_op(), self.val_x, self.val_y, is_train=False)
                total_val_preds.append(valid_preds)
                total_val_loss.append(valid_loss)
            
            val_labels = self.val_y
            val_stacked_preds = np.reshape(np.concatenate(total_val_preds, 0), [self.num_sampling, self.val_x.shape[0], self.val_y.shape[1]])
            val_preds = np.mean(val_stacked_preds, axis=0)
            val_loss = np.mean(total_val_loss, axis=0)
            roc, valid_auc = ROC_AUC(val_preds, val_labels)
            val_preds = val_preds >= 0.5
            val_acc = accuracy(val_preds, val_labels)
            print(" [*] Epoch: %d, Validation loss: %.4f, Validation AUC: %.4f, Validation ACC: %.4f" % (i_+1, valid_loss, valid_auc, val_acc))
            
            total_eval_preds=[]
            total_eval_loss=[]
            for sample in range(self.num_sampling):
                eval_loss, eval_auc, eval_acc, eval_preds = self.run_epoch(tf.no_op(), self.eval_x, self.eval_y, False)
                total_eval_preds.append(eval_preds)
                total_eval_loss.append(eval_loss)
            
            eval_labels = self.eval_y
            eval_stacked_preds = np.reshape(np.concatenate(total_eval_preds, 0), [self.num_sampling, self.eval_x.shape[0], self.eval_y.shape[1]])
            eval_preds = np.mean(eval_stacked_preds, axis=0)
            eval_loss = np.mean(total_eval_loss, axis=0)
            roc, eval_auc = ROC_AUC(eval_preds, eval_labels)
            eval_preds = eval_preds >= 0.5
            eval_acc = accuracy(eval_preds, eval_labels)
            print(" [*] Epoch: %d, Evaluation loss: %.4f, Evaluation AUC: %.4f, Evaluation ACC: %.4f" % (i_+1, eval_loss, eval_auc, eval_acc))
            print("=======================================================================================")

            if (i_+1)%self.save_iter == 0:
                save_path = saver.save(self.sess, 'checkpoints_UA/'+self.task+'_'+str(i_+1)+'_%.4f'%(eval_loss)+'_%.4f'%(eval_auc)+'.ckpt')
      
