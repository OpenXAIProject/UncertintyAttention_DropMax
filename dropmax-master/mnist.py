import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def mnist_input(path, nlist):
    mnist = input_data.read_data_sets(path, one_hot=True, validation_size=0)
    x, y = mnist.train.images, mnist.train.labels
    y_ = np.argmax(y, axis=1)

    xtr = [x[y_==k][:nlist[k],:] for k in range(10)]
    ytr = [y[y_==k][:nlist[k],:] for k in range(10)]
    xtr, ytr = np.concatenate(xtr, axis=0), np.concatenate(ytr, axis=0)

    xva = [x[y_==k][nlist[k]:2*nlist[k],:] for k in range(10)]
    yva = [y[y_==k][nlist[k]:2*nlist[k],:] for k in range(10)]
    xva, yva = np.concatenate(xva, axis=0), np.concatenate(yva, axis=0)

    xte, yte = mnist.test.images, mnist.test.labels
    return xtr, ytr, xva, yva, xte, yte
