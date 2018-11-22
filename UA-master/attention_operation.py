import tensorflow as tf
import numpy as np
import time
import math
from six.moves import xrange
import scipy.io
import os
import collections
from sklearn.metrics import roc_curve, auc

def attention_op(str_id, rnn_outputs, hidden_units, embed_size, steps):
	with tf.variable_scope(str_id+'_'):
		if str_id == 'alpha':
			p_att_shape = [hidden_units, 1]
		elif str_id == 'beta':
			p_att_shape = [hidden_units, embed_size]
		else:
			raise ValueError('You must re-check the attention id. required to \'alpha\' or \'beta\'')

	#Create MU
	with tf.variable_scope(str_id+'MU'):
		mu_w = tf.Variable(tf.random_normal(p_att_shape, stddev=0.01), name='_mu')
		mu_b = tf.Variable(tf.zeros(p_att_shape[1], name='_mu'))
		mu =[]
		for _i in range(steps):
			mu_tmp = tf.matmul(rnn_outputs[:, _i, :], mu_w) + mu_b
			mu.append(mu_tmp)
		mu = tf.reshape(tf.concat(mu, 1), [-1, steps, p_att_shape[1]])

	#Create sigma
	with tf.variable_scope(str_id+'SIGMA'):
		sigma_w = tf.Variable(tf.random_normal(p_att_shape, stddev=0.01), name='sigma_weight')
		sigma_b = tf.Variable(tf.zeros(p_att_shape[1], name='sigma_bias'))
		sigma=[]
		for _k in range(steps):
			sigma_tmp = tf.matmul(rnn_outputs[:, _k, :], sigma_w) + sigma_b
			sigma.append(sigma_tmp)
		sigma = tf.reshape(tf.concat(sigma, 1), [-1, steps, p_att_shape[1]])
		sigma = tf.nn.softplus(sigma)

	distribution = tf.distributions.Normal(loc=mu, scale=sigma)
	att = distribution.sample([1])
	att = tf.squeeze(att, 0)

	if str_id == 'alpha':
		squashed_att = tf.nn.softmax(att, 1)
		print('Done with generating alpha attention.')
	elif str_id == 'beta':
		squashed_att = tf.nn.tanh(att)
		print('Done with generating beta attention.')
	else:
		raise ValueError('You must re-check the attention id. required to \'alpha\' or \'beta\'')
        return squashed_att

