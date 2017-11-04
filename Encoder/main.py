# TensorFlow version of NIPS2016 soundnet
# Required package: librosa: A python package for music and audio analysis.
# $ pip install librosa

from ops import batch_norm, conv2d, relu, maxpool
from util import preprocess, load_from_list, load_audio
from model import Model
from glob import glob

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
    'epoch': 200,
    'eps': 1e-5,
    'learning_rate': 1e-3,
    'beta1': 0.9,
    'load_size': 22050 * 4,
    'sample_rate': 22050,
    'name_scope': 'SoundNet',
    'phase': 'train',
    'dataset_name': 'ESC50',
    'subname': 'mp3',
    'checkpoint_dir': 'checkpoint',
    'dump_dir': 'output',
    'model_dir': None,
    'param_g_dir': './models/sound5.npy',
}


class Model():
    def __init__(self, session, config=local_config, param_G=None):
        self.sess = session
        self.config = config
        self.param_G = param_G
        self.g_step = tf.Variable(0, trainable=False)
        self.counter = 0
        self.model()

    def model(self):
        # Placeholder
        self.sound_input_placeholder = tf.placeholder(tf.float32,
                                                      shape=[self.config['batch_size'], None, 1,
                                                             1])  # batch x h x w x channel
        self.object_dist = tf.placeholder(tf.float32,
                                          shape=[self.config['batch_size'], None, 1000])  # batch x h x w x channel
        self.scene_dist = tf.placeholder(tf.float32,
                                         shape=[self.config['batch_size'], None, 401])  # batch x h x w x channel

        self.is_train = True
        self.keep_prob_fc5 = tf.placeholder(tf.float32)
        self.keep_prob_fc6 = tf.placeholder(tf.float32)

        # Generator
        self.add_generator(name_scope=self.config['name_scope'])

        #compute loss
        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.add_to_collection('losses', cross_entropy_mean)
        self.loss_op = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # KL Divergence
        self.object_loss = self.KL_divergence(self.layers[16], self.object_dist, name_scope='KL_Div_object')
        self.scene_loss = self.KL_divergence(self.layers[17], self.scene_dist, name_scope='KL_Div_scene')
        self.loss = self.object_loss + self.scene_loss

        #loss
        # Summary
        self.loss_sum = tf.summary.scalar("g_loss", self.loss)
        self.g_sum = tf.summary.merge([self.loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # variable collection
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.config['name_scope'])

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=12,
                                    max_to_keep=5,
                                    restore_sequentially=True)

        # Optimizer and summary
        self.g_optim = tf.train.AdamOptimizer(self.config['learning_rate'], beta1=self.config['beta1']) \
            .minimize(self.loss, var_list=(self.g_vars), global_step=self.g_step)

        # Initialize
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Load checkpoint
        if self.load(self.config['checkpoint_dir']):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def FC(self, input, inp_neurons, out_neurons):
        # W = tf.get_variable('w',[inp_neurons,out_neurons], tf.float32, tf.random_normal_initializer(0.0, 0.02))
        W = tf.get_variable('w', [inp_neurons, out_neurons], tf.float32, tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [out_neurons], initializer=tf.constant_initializer(0.0))

        weight_decay = tf.multiply(tf.nn.l2_loss(W), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

        return tf.matmul(input, W) + b

    def add_generator(self, name_scope='SoundNet'):
        with tf.variable_scope(name_scope) as scope:
            self.layers = {}

            # Stream one: conv1 ~ conv7
            self.layers[1] = conv2d(self.sound_input_placeholder, 1, 32, k_h=64, d_h=2, p_h=32, name_scope='conv1')
            self.layers[2] = batch_norm(self.layers[1], 32, self.config['eps'], name_scope='conv1')
            self.layers[3] = relu(self.layers[2], name_scope='conv1')
            self.layers[4] = maxpool(self.layers[3], k_h=8, d_h=8, name_scope='conv1')
            print('Conv1 {}'.format(self.layers[4].shape))

            self.layers[5] = conv2d(self.layers[4], 32, 64, k_h=32, d_h=2, p_h=16, name_scope='conv2')
            self.layers[6] = batch_norm(self.layers[5], 64, self.config['eps'], name_scope='conv2')
            self.layers[7] = relu(self.layers[6], name_scope='conv2')
            self.layers[8] = maxpool(self.layers[7], k_h=8, d_h=8, name_scope='conv2')
            print('Conv2 {}'.format(self.layers[8].shape))

            self.layers[9] = conv2d(self.layers[8], 64, 128, k_h=16, d_h=2, p_h=8, name_scope='conv3')
            self.layers[10] = batch_norm(self.layers[9], 128, self.config['eps'], name_scope='conv3')
            self.layers[11] = relu(self.layers[10], name_scope='conv3')
            self.layers[12] = maxpool(self.layers[11], k_h=8, d_h=8, name_scope='conv3')
            print('Conv3 {}'.format(self.layers[12].shape))

            self.layers[13] = conv2d(self.layers[12], 128, 256, k_h=8, d_h=2, p_h=4, name_scope='conv4')
            self.layers[14] = batch_norm(self.layers[13], 256, self.config['eps'], name_scope='conv4')
            self.layers[15] = relu(self.layers[14], name_scope='conv4')
            # Split one: conv8, conv8_2
            # NOTE: here we use a padding of 2 to skip an unknown error
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L45
            # self.flat = tf.contrib.layers.flatten(self.layers[15])
            # print('Flat final {}'.format(self.flat.shape))
            self.layers[16] = conv2d(self.layers[15], 256, 1000, k_h=16, d_h=12, p_h=4, name_scope='conv5')
            self.layers[17] = conv2d(self.layers[15], 256, 401, k_h=16, d_h=12, p_h=4, name_scope='conv5_2')
            self.layers[18] = tf.contrib.layers.flatten(self.layers[15])
            self.layers[19] = self.FC(self.layers[18], self.layers[18].get_shape()[1], 1024)
            self.layers[20] = tf.nn.relu(self.layers[19])
            if self.is_train:
                self.drop_out5 = tf.nn.dropout(self.layers[20], self.keep_prob_fc5)
            else:
                self.drop_out5 = self.layers[20]

            self.layers[21] = self.FC(self.drop_out5, self.drop_out5.get_shape()[1], 128)
            self.layers[22] = tf.nn.relu(self.layers[21])
            if self.is_train:
                self.drop_out6 = tf.nn.dropout(self.layers[22], self.keep_prob_fc6)
            else:
                self.drop_out6 = self.layers[22]


    def train(self):
        """Train SoundNet"""

        start_time = time.time()

        # Data info
        data = glob('./data/*.{}'.format(self.config['subname']))
        batch_idxs = min(len(data), self.config['train_size']) // self.config['batch_size']
        for epoch in xrange(self.counter // batch_idxs, self.config['epoch']):

            for idx in xrange(self.counter % batch_idxs, batch_idxs):

                # By default, librosa will resample the signal to 22050Hz. And range in (-1., 1.)
                sound_sample = load_from_list(
                    data[idx * self.config['batch_size']:(idx + 1) * self.config['batch_size']], self.config)

                # Update G network
                # NOTE: Here we still use dummy random distribution for scene and objects
                _, summary_str, l_scn, l_obj = self.sess.run(
                    [self.g_optim, self.g_sum, self.scene_loss, self.object_loss],
                    feed_dict={self.sound_input_placeholder: sound_sample, \
                               self.scene_dist: np.random.randint(2, size=(1, 1, 401)), \
                               self.object_dist: np.random.randint(2, size=(1, 1, 1000))})
                self.writer.add_summary(summary_str, self.counter)

                print("[Epoch {}] {}/{} | Time: {} | scene_loss: {} | obj_loss: {}".format(epoch, idx, batch_idxs,
                                                                                           time.time() - start_time,
                                                                                           l_scn, l_obj))

                if np.mod(self.counter, 1000) == 1000 - 1:
                    self.save(self.config['checkpoint_dir'], self.counter)

                self.counter += 1

    #########################
    #          Loss         #
    #########################
    # Adapt the answer here: http://stackoverflow.com/questions/41863814/kl-divergence-in-tensorflow
    def KL_divergence(self, dist_a, dist_b, name_scope='KL_Div'):
        return tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=dist_a, labels=dist_b))

    #########################
    #       Save/Load       #
    #########################
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
            with tf.variable_scope(self.config['name_scope'] + '/' + key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        self.sess.run(var.assign(data_dict[key][subkey]))
                        print('Assign pretrain model {} to {}'.format(subkey, key))
                    except:
                        print('Ignore {}'.format(key))

        self.param_G.clear()
        return True


def main():
    args = parse_args()
    local_config['phase'] = args.phase

    # Setup visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Make path
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    # Load pre-trained model
    param_G = np.load(local_config['param_g_dir'], encoding='latin1').item() \
        if args.phase in ['finetune', 'extract'] \
        else None

    # Init. Session
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as session:
        # Build model
        model = Model(session, config=local_config, param_G=param_G)

        if args.phase in ['train', 'finetune']:
            # Training phase
            model.train()
        elif args.phase == 'extract':
            # import when we need
            from extract_feat import extract_feat

            # Feature extractor
            # sound_sample = np.reshape(np.load('./data/demo.npy', encoding='latin1'), [local_config['batch_size'], -1, 1, 1])

            import librosa
            audio_path = './data/demo.mp3'
            sound_sample, _ = load_audio(audio_path)
            sound_sample = preprocess(sound_sample, config=local_config)

            output = extract_feat(model, sound_sample, args)


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='SoundNet')

    parser.add_argument('-o', '--outpath', dest='outpath', help='output feature path. e.g., [output]', default='output')

    parser.add_argument('-p', '--phase', dest='phase', help='demo or extract feature. e.g., [train, finetune, extract]',
                        default='finetune')

    parser.add_argument('-m', '--layer', dest='layer_min', help='start from which feature layer. e.g., [1]', type=int,
                        default=1)

    parser.add_argument('-x', dest='layer_max', help='end at which feature layer. e.g., [24]', type=int, default=None)

    parser.add_argument('-c', '--cuda', dest='cuda_device', help='which cuda device to use. e.g., [0]', default='0')

    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('-s', '--save', dest='is_save', help='Turn on save mode. [False(default), True]',
                                action='store_true')
    parser.set_defaults(is_save=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
