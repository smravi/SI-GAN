from ops import batch_norm, conv2d, relu, maxpool
# from model import Model



import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import os


# Make xrange compatible in both Python 2, 3
try:
    xrange
except NameError:
    xrange = range

local_config = {
            'batch_size': 1,
            'train_size': np.inf,
            'epoch': 5,
            'eps': 1e-5,
            'learning_rate': 1e-3,
            'beta1': 0.9,
            'load_size': 22050*4,
            'sample_rate': 22050,
            'name_scope': 'SoundNet',
            'phase': 'train',
            'dataset_name': 'ESC50',
            'subname': 'mp3',
            'checkpoint_dir': 'checkpoint',
            'dump_dir': 'output',
            'model_dir': None,
            'param_g_dir': './models/sound5.npy',
            'fcparam_g_dir': './models/sound5_FC.npy'
            }


class Model():
    def __init__(self, session, config=local_config, param_G=None,fcparam_G=None):
        self.sess           = session
        self.config         = config
        self.param_G        = param_G
        self.fcparam_G      = fcparam_G
        self.g_step         = tf.Variable(0, trainable=False)
        self.counter        = 0
        self.model()
 

    def model(self):        
        self.sound_input_placeholder = tf.placeholder(tf.float32,
                shape=[None, 220500, 1, 1]) # batch x h x w x channel

        self.add_Encoder(name_scope=self.config['name_scope'])
        
        # Initialize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        self.load_from_npy()


    def add_Encoder(self, name_scope='SoundNet',is_train = True):
        with tf.variable_scope(name_scope) as scope:
            self.layers = {}
            self.layers[1] = conv2d(self.sound_input_placeholder, 1, 32, k_h=64, d_h=2, p_h=32, name_scope='conv1')
            self.layers[2] = batch_norm(self.layers[1], 32, local_config['eps'], name_scope='conv1')
            self.layers[3] = relu(self.layers[2], name_scope='conv1')
            self.layers[4] = maxpool(self.layers[3], k_h=8, d_h=8, name_scope='conv1')
            print('Conv1 {}'.format(self.layers[4].shape))

            self.layers[5] = conv2d(self.layers[4], 32, 64, k_h=32, d_h=2, p_h=16, name_scope='conv2')
            self.layers[6] = batch_norm(self.layers[5], 64, local_config['eps'], name_scope='conv2')
            self.layers[7] = relu(self.layers[6], name_scope='conv2')
            self.layers[8] = maxpool(self.layers[7], k_h=8, d_h=8, name_scope='conv2')
            print('Conv2 {}'.format(self.layers[8].shape))

            self.layers[9] = conv2d(self.layers[8], 64, 128, k_h=16, d_h=2, p_h=8, name_scope='conv3')
            self.layers[10] = batch_norm(self.layers[9], 128, local_config['eps'], name_scope='conv3')
            self.layers[11] = relu(self.layers[10], name_scope='conv3')
            self.layers[12] = maxpool(self.layers[11], k_h=8, d_h=8, name_scope='conv3')
            print('Conv3 {}'.format(self.layers[12].shape))

            self.layers[13] = conv2d(self.layers[12], 128, 256, k_h=8, d_h=2, p_h=4, name_scope='conv4')
            self.layers[14] = batch_norm(self.layers[13], 256, local_config['eps'], name_scope='conv4')
            self.layers[15] = relu(self.layers[14], name_scope='conv4')
            print('Conv4 {}'.format(self.layers[14].shape))

            self.layers[17] = conv2d(self.layers[15], 256, 401, k_h=16, d_h=12, p_h=4, name_scope='conv5_2')
            print('Conv5 {}'.format(self.layers[17].shape))

            self.layers[18] = tf.contrib.layers.flatten(self.layers[17])
            print('Flat {}'.format(self.layers[18].shape))



    @property
    def get_model_dir(self):
        if self.config['model_dir'] is None:
            return "{}_{}".format(
                self.config['dataset_name'], self.config['batch_size'])
        else:
            return self.config['model_dir']
    

    def load(self, ckpt_dir='checkpoint'):
        return self.load_from_ckpt(ckpt_dir) if self.param_G is None \
        else self.load_from_npy()


    def save(self, checkpoint_dir, step):
        """ Checkpoint saver """
        model_name = "SoundNet.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        if not os.path.exists(checkpoint_dir):
            
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load_from_ckpt(self, checkpoint_dir='checkpoint'):
        """ Checkpoint loader """
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            self.counter = int(ckpt_name.rsplit('-', 1)[-1])
            print(" [*] Start counter from {}".format(self.counter))
            return True
        else:
            print(" [*] Failed to find a checkpoint under {}".format(checkpoint_dir))
            return False


    def load_from_npy(self):
        if self.param_G is None: return False
        data_dict = self.param_G
        for key in data_dict:
            if(key=='conv5'):
                continue
            with tf.variable_scope(self.config['name_scope'] + '/'+ key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        self.sess.run(var.assign(data_dict[key][subkey]))
                        print('Assign pretrain model {} to {}'.format(subkey, key))
                    except:
                        print('Ignore {}'.format(key))
        # self.param_G.clear()
        return True
