import tensorflow as tf
from model_UA import *
import tensorflow as tf
import sys

config = {}

# Data info
config['task'] = 'UA_source_code'
config['num_features'] = 0
config['steps'] = 0

# Model info
config['max_epoch'] = 100
config['num_layers'] = 1
config['hidden_units'] = 33
config['embed_size'] = 33
config['lr'] = 1e-4
config['batch_size'] = 100
config['save_iter'] = 20
config['num_sampling'] = 30
config['lamb'] = 0.002

def main():
    
    path = 'physionet_dataset/1_'

    train_x = np.load(path + 'train_x.npy')
    train_y = np.load(path + 'train_y.npy')
    val_x = np.load(path + 'val_x.npy')
    val_y = np.load(path + 'val_y.npy')
    eval_x = np.load(path + 'eval_x.npy')
    eval_y = np.load(path + 'eval_y.npy') 
    num_features = train_x.shape[2]                 
    steps = train_x.shape[1]                

    print('shape of train_x:', train_x.shape)

    config['num_features'] = num_features
    config['steps'] = steps
    config['train_x'] = train_x
    config['train_y'] = train_y
    config['val_x'] = val_x
    config['val_y'] = val_y
    config['eval_x'] = eval_x
    config['eval_y'] = eval_y

    #GPU Option
    gpu_usage = 0.95
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    config['sess'] = sess


    with tf.Session() as sess:

        model = UA(config)
        model.build_model()
        model.run()


if __name__ == '__main__':
    main()
