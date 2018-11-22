from layers import *

def base_softmax(x, y, training, name='base_softmax', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = conv(x, 20, 5, name=name+'/conv1', reuse=reuse)
    x = relu(x)
    x = pool(x, name=name+'/pool1')
    x = conv(x, 50, 5, name=name+'/conv2', reuse=reuse)
    x = relu(x)
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    h = dense(x, 500, activation=relu, name=name+'/dense', reuse=reuse)
    o = dense(h, 10, name=name+'/logits', reuse=reuse)

    net = {}
    net['cent'] = cross_entropy(exp(o), y)
    net['acc'] = accuracy(exp(o), y)
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = all_vars
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    return net

def dropmax(x, y, training, name='dropmax', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = conv(x, 20, 5, name=name+'/conv1', reuse=reuse)
    x = relu(x)
    x = pool(x, name=name+'/pool1')
    x = conv(x, 50, 5, name=name+'/conv2', reuse=reuse)
    x = relu(x)
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    h = dense(x, 500, activation=relu, name=name+'/dense', reuse=reuse)

    # dropmax branches
    o = dense(h, 10, name=name+'/logits', reuse=reuse)
    ph = dense(h, 10, name=name+'/ph_branch', reuse=reuse)
    rh = dense(h, 10, name=name+'/rh_branch', reuse=reuse)
    qh = tf.stop_gradient(ph) + rh

    p = sigmoid(ph)
    r = sigmoid(rh)
    q = sigmoid(qh)

    # sampling the dropout masks
    z = genmask(q, y)

    net = {}
    net['cent'] = cross_entropy((z+eps)*exp(o), y)
    net['acc'] = accuracy((p+eps)*exp(o), y)
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = all_vars
    net['wd'] = weight_decay(1e-4,
            var_list=[v for v in all_vars if 'branch' not in v.name])

    # dropmax modules
    net['kl'] = kl_divergence(p, q, y)
    net['aux'] = auxloss(r, y)
    net['ent'] = entropy(p)
    return net
