from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from lenet import base_softmax, dropmax
from accumulator import Accumulator
from mnist import mnist_input
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mnist_path', type=str, default='./mnist')
parser.add_argument('--model', type=str, default='softmax')
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--gpu_num', type=int, default=0)
args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

savedir = './results/%s'%args.model if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

bs = args.batch_size
N = args.N

xtr, ytr, xva, yva, xte, yte = mnist_input(args.mnist_path, [N]*10)
n_train_batches, n_val_batches, n_test_batches = N*10/bs, N*10/bs, 10000/bs

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

if args.model == 'softmax':
    model = base_softmax
elif args.model == 'dropmax':
    model = dropmax
else:
    raise ValueError('Invalid model %s' % args.model)
net = model(x, y, True)
tnet = model(x, y, False, reuse=True)

def train():
    if args.model == 'softmax':
        loss = net['cent'] + net['wd']
    else:
        loss = net['cent'] + net['wd'] + net['kl'] + net['aux'] + net['ent']

    global_step = tf.train.get_or_create_global_step()
    lr_step = n_train_batches*args.n_epochs/3
    lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
            [lr_step, lr_step*2], [1e-3, 1e-4, 1e-5])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(net['weights'])
    logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_logger = Accumulator('cent', 'acc')
    train_to_run = [train_op, net['cent'], net['acc']]
    val_logger = Accumulator('cent', 'acc')
    val_to_run = [tnet['cent'], tnet['acc']]

    for i in range(args.n_epochs):
        # shuffle the training data every epoch
        xytr = np.concatenate((xtr, ytr), axis=1)
        np.random.shuffle(xytr)
        xtr_, ytr_ = xytr[:,:784], xytr[:,784:]

        line = 'Epoch %d start, learning rate %f' % (i+1, sess.run(lr))
        print(line)
        logfile.write(line + '\n')
        train_logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            bx, by = xtr_[j*bs:(j+1)*bs,:], ytr_[j*bs:(j+1)*bs,:]
            train_logger.accum(sess.run(train_to_run, {x:bx, y:by}))
        train_logger.print_(header='train', epoch=i+1,
                time=time.time()-start, logfile=logfile)

        val_logger.clear()
        for j in range(n_val_batches):
            bx, by = xva[j*bs:(j+1)*bs,:], yva[j*bs:(j+1)*bs,:]
            val_logger.accum(sess.run(val_to_run, {x:bx, y:by}))
        val_logger.print_(header='val', epoch=i+1,
                time=time.time()-start, logfile=logfile)
        print()
        logfile.write('\n')

    logfile.close()
    saver.save(sess, os.path.join(savedir, 'model'))

def test():
    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights'])
    saver.restore(sess, os.path.join(savedir, 'model'))

    logfile = open(os.path.join(savedir, 'test.log'), 'w', 0)
    logger = Accumulator('cent', 'acc')
    logger.accum(sess.run([tnet['cent'], tnet['acc']], {x:xte, y:yte}))
    logger.print_(header='test', logfile=logfile)
    logfile.close()

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode %s' % args.mode)
