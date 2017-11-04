import tensorflow as tf
import numpy as np
import os

# Final Working Model with 70% accuracy

num_training = 300
num_validation = 50
num_test = 50

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
    'param_g_dir': '../models/sound5.npy',
}


def conv2d(prev_layer, in_ch, out_ch, k_h=1, k_w=1, d_h=1, d_w=1, p_h=0, p_w=0, pad='VALID', name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        # h x w x input_channel x output_channel
        w_conv = tf.get_variable('weights', [k_h, k_w, in_ch, out_ch],
                                 initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
        b_conv = tf.get_variable('biases', [out_ch],
                                 initializer=tf.constant_initializer(0.0))

        padded_input = tf.pad(prev_layer, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], "CONSTANT") if pad == 'VALID' \
            else prev_layer

        output = tf.nn.conv2d(padded_input, w_conv,
                              [1, d_h, d_w, 1], padding=pad, name='z') + b_conv

        return output


def batch_norm(prev_layer, out_ch, eps, name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        # mu_conv, var_conv = tf.nn.moments(prev_layer, [0, 1, 2], keep_dims=False)
        mu_conv = tf.get_variable('mean', [out_ch],
                                  initializer=tf.constant_initializer(0))
        var_conv = tf.get_variable('var', [out_ch],
                                   initializer=tf.constant_initializer(1))
        gamma_conv = tf.get_variable('gamma', [out_ch],
                                     initializer=tf.constant_initializer(1))
        beta_conv = tf.get_variable('beta', [out_ch],
                                    initializer=tf.constant_initializer(0))
        output = tf.nn.batch_normalization(prev_layer, mu_conv,
                                           var_conv, beta_conv, gamma_conv, eps, name='batch_norm')

        return output


def relu(prev_layer, name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        return tf.nn.relu(prev_layer, name='a')


def maxpool(prev_layer, k_h=1, k_w=1, d_h=1, d_w=1, name_scope='conv'):
    with tf.variable_scope(name_scope) as scope:
        return tf.nn.max_pool(prev_layer,
                              [1, k_h, k_w, 1], [1, d_h, d_w, 1], padding='VALID', name='maxpool')


class SoundEncoder(object):
    def __init__(self):
        self.num_epoch = local_config['epoch']
        self.batch_size = local_config['batch_size']
        self.param_G = np.load(local_config['param_g_dir'], encoding='latin1').item()
        # Load checkpoint
        self._build_model()
        if self.load(local_config['checkpoint_dir']):
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

    def Normalization(self, input):
        return tf.nn.local_response_normalization(input,
                                                  alpha=0.001 / 9.0,
                                                  beta=0.75,
                                                  depth_radius=4,
                                                  bias=1.0)

    @property
    def get_model_dir(self):
        if local_config['model_dir'] is None:
            return "{}_{}".format(
                local_config['dataset_name'], local_config['batch_size'])
        else:
            return local_config['model_dir']

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
            with tf.variable_scope(local_config['name_scope'] + '/' + key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        self.sess.run(var.assign(data_dict[key][subkey]))
                        print('Assign pretrain model {} to {}'.format(subkey, key))
                    except:
                        print('Ignore {}'.format(key))

        self.param_G.clear()
        return True

    def _model(self):
        print('-' * 5 + '  Sample model  ' + '-' * 5)

        print('intput layer: ' + str(self.X.get_shape()))

        self.layers = {}

        # Stream one: conv1 ~ conv7
        self.layers[1] = conv2d(self.X, 1, 32, k_h=64, d_h=2, p_h=32, name_scope='conv1')
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
        print('Conv3 {}'.format(self.layers[14].shape))
        # Split one: conv8, conv8_2
        # NOTE: here we use a padding of 2 to skip an unknown error
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L45
        # self.flat = tf.contrib.layers.flatten(self.layers[15])
        # print('Flat final {}'.format(self.flat.shape))
        self.layers[16] = conv2d(self.layers[15], 256, 1000, k_h=16, d_h=12, p_h=4, name_scope='conv5')
        self.layers[17] = conv2d(self.layers[15], 256, 401, k_h=16, d_h=12, p_h=4, name_scope='conv5_2')
        self.layers[18] = tf.contrib.layers.flatten(self.layers[15])
        self.layers[19] = tf.layers.dense(self.layers[18],1024, activation=tf.nn.relu, name="fc1")
        self.layers[20] = tf.layers.dense(self.layers[19], 128, activation=tf.nn.relu, name="fc2")
        self.layers[21] = tf.layers.dense(self.layers[20], 10, None,name="fc3")

        # self.layers[19] = self.FC(self.layers[18], self.layers[18].get_shape()[1], 1024)
        # self.layers[20] = tf.nn.relu(self.layers[19])
        # if self.is_train:
        #     self.drop_out5 = tf.nn.dropout(self.layers[20], self.keep_prob_fc5)
        # else:
        #     self.drop_out5 = self.layers[20]
        #
        # self.layers[21] = self.FC(self.drop_out5, self.drop_out5.get_shape()[1], 128)
        # self.layers[22] = tf.nn.relu(self.layers[21])
        # if self.is_train:
        #     self.drop_out6 = tf.nn.dropout(self.layers[22], self.keep_prob_fc6)
        # else:
        #     self.drop_out6 = self.layers[22]
        #
        # with tf.variable_scope('fc7'):
        #
        #     self.fc7 = self.FC(self.drop_out6, self.drop_out6.get_shape()[1], 10)
        #     print('fc7 layer: ' + str(self.fc7.get_shape()))
        #
        # # Return the last layer
        # return self.fc7
        return self.layers[21]

    def _input_ops(self):
        # Placeholders
        self.X = tf.placeholder(tf.float32, [None,220500, 1, 1])
        self.Y = tf.placeholder(tf.int64, [None])

        self.is_train = True
        self.keep_prob_fc5 = tf.placeholder(tf.float32)
        self.keep_prob_fc6 = tf.placeholder(tf.float32)

    def _build_optimizer(self):
        # Adam optimizer 'self.train_op' that minimizes 'self.loss_op'
        # Optimizer and summary
        t_vars=tf.global_variables()
        all_vars=[]
        for name in ['fc1','fc2','fc3']:
            d_vars = [var for var in t_vars if name in var.name]
            all_vars.append(d_vars)
        self.global_step = tf.Variable(0, trainable=False)
        self.initial_lr = local_config['learning_rate']
        # self.exp_decay = tf.train.exponential_decay(self.initial_lr, self.global_step, 500, 0.96)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.initial_lr, beta1=local_config['beta1']).minimize(
            self.loss_op,
            global_step=self.global_step,
            var_list=all_vars)

    def _loss(self, labels, logits):

        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.add_to_collection('losses', cross_entropy_mean)
        # self.loss_op = tf.add_n(tf.get_collection('losses'), name='total_loss')
        self.loss_op = cross_entropy_mean

    def _build_model(self):
        # Define input variables
        self._input_ops()

        # Convert Y to one-hot vector
        labels = tf.one_hot(self.Y, 10)

        # Build a model and get logits
        logits = self._model()

        # Compute loss
        self._loss(labels, logits)

        # Build optimizer
        self._build_optimizer()

        # Compute accuracy
        predict = tf.argmax(logits, 1)
        correct = tf.equal(predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)

        self.is_train = True
        self.log_step = 5
        for epoch in range(self.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.batch_size):
                X_ = X_train[i * self.batch_size:(i + 1) * self.batch_size][:]
                Y_ = Y_train[i * self.batch_size:(i + 1) * self.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.keep_prob_fc5: 0.7, self.keep_prob_fc6: 0.8}
                fetches = [self.train_op, self.loss_op, self.accuracy_op]

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)
                if step % self.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                          (step, loss, accuracy))
                step += 1

            # Print validation results
            self.is_train = False
            print('validation for epoch %d' % epoch)
            val_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))
            self.is_train = True

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size][:]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]

            feed_dict = {self.X: X_, self.Y: Y_, self.keep_prob_fc5: 0.7, self.keep_prob_fc6: 0.8}
            accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_iter += 1
        return eval_accuracy / eval_iter


def load_data():
    # load the data

    sound_samples = np.load('../data/shruthi_x.npy', encoding='latin1')
    print(sound_samples.shape)
    sound_labels = np.load('../data/shruthi_y.npy', encoding='latin1').reshape(-1)
    print(sound_labels.shape)
    X_train = sound_samples[:300]
    X_val = sound_samples[300:350]
    X_test = sound_samples[350:400]
    print(X_train.shape)

    Y_train = sound_labels[:300]
    Y_val = sound_labels[300:350]
    Y_test = sound_labels[350:400]
    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'X_test': X_test,
        'Y_test': Y_test
    }



def train_encoder():
    # train the enoder with the necessary params

    tf.reset_default_graph()
    sess = tf.Session()
    data = load_data()
    model = SoundEncoder()

    model.train(sess, data['X_train'], data['Y_train'], data['X_val'], data['Y_val'])
    model.is_train = False
    accuracy = model.evaluate(sess, data['X_test'], data['Y_test'])
    print('***** test accuracy: %.3f' % accuracy)

    # Save your model
    saver = tf.train.Saver()
    model_path = saver.save(sess, "./S2I.ckpt")
    print("Model saved in %s" % model_path)

    sess.close()


if __name__ == '__main__':
    train_encoder()