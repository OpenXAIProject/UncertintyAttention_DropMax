import tensorflow as tf
import numpy as np

exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1-x)
sigmoid = tf.nn.sigmoid
softmax = tf.nn.softmax
relu = tf.nn.relu
tau = 0.1
eps = 1e-20

dense = tf.layers.dense
flatten = tf.contrib.layers.flatten

# network components
def conv(x, filters, kernel_size=3, strides=1, **kwargs):
    return tf.layers.conv2d(x, filters, kernel_size, strides,
            data_format='channels_first', **kwargs)

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])

# training modules
def cross_entropy(expo, y):
    denom = log(tf.reduce_sum(expo, axis=1))
    numer = log(tf.reduce_sum(tf.multiply(expo, y), axis=1))
    return -tf.reduce_mean(numer - denom)

def accuracy(expo, y):
    correct = tf.equal(tf.argmax(expo, 1), tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def weight_decay(decay, var_list=None):
    var_list = tf.trainable_variables() if var_list is None else var_list
    return decay*tf.add_n([tf.nn.l2_loss(var) for var in var_list])

# dropmax modules
def dist_unif(p):
    return tf.contrib.distributions.Uniform(tf.zeros_like(p), tf.ones_like(p))

def genmask(p, y):
    u = dist_unif(p).sample()
    z = sigmoid(1/tau * (logit(p) + logit(u)))
    return tf.where(tf.equal(y,1), tf.ones_like(z), z)

def kl_divergence(p, q, y):
    target = -log(p)
    nontarget = q*(log(q)-log(p)) + (1-q)*(log(1-q)-log(1-p))
    kl = tf.where(tf.equal(y,1), target, nontarget)
    return tf.reduce_sum(tf.reduce_mean(kl,0))

def auxloss(r, y):
    target = -log(r)
    nontarget = -log(1-r)
    aux = tf.where(tf.equal(y,1), target, nontarget)
    return tf.reduce_sum(tf.reduce_mean(aux,0))

def entropy(p):
    ent = p*log(p) + (1-p)*log(1-p)
    return tf.reduce_sum(tf.reduce_mean(ent,0))
