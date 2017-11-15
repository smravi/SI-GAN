""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re

from utils import *
from dc_model import *
import dc_model

# Set the path for Sound Encoder
# import sys
# sys.path.append('./Encoder/')

# Import Soundnet
from Encoder import *

class WGAN:
    def __init__(self):
        self.sess           = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        #self.config         = config
        self.param_G        = np.load('./models/sound5.npy', encoding='latin1').item()
        self.sound_config ={
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
            }
        self.EncodeModel()

        #For saving checkpoints
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, 
                                    max_to_keep=5, 
                                    restore_sequentially=True)

    def EncodeModel(self):
        self.sound_config['name_scope'] = 'Soundnet1'
        self.sound_encoder1 = Model(self.sess, config=self.sound_config,param_G=self.param_G,fcparam_G=None)
        self.sound_config['name_scope'] = 'Soundnet2'
        self.sound_encoder2 = Model(self.sess, config=self.sound_config, param_G=self.param_G,fcparam_G=None)

    def load_data(self):
        datapath = './data/'
        sound_train = np.load(datapath+'esc_44_sound.npy')
        image_train = np.load(datapath+'esc_44_image.npy')
        return sound_train, image_train

    def _get_optimizer(self, optimizer_name, learning_rate, optimizer_param):
        self.learning_rate = learning_rate
        if optimizer_name == "Adam":
            return tf.train.AdamOptimizer(learning_rate, beta1=optimizer_param)
        elif optimizer_name == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate, decay=optimizer_param)
        else:
            raise ValueError("Unknown optimizer %s" % optimizer_name)

    def _gradient_minimize(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        for grad, var in grads:
            add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    def load_from_ckpt(self, checkpoint_dir='checkpoint'):
        """ Checkpoint loader """
        print(" [*] Reading checkpoints...")

        # checkpoint_dir = os.path.join(checkpoint_dir, self.get_model_dir)

        path = checkpoint_dir+"/sigan"+".ckpt"
        if(os.path.exists(checkpoint_dir+"/sigan"+".ckpt")):
            try:
                self.saver.restore(self.sess, checkpoint_dir+"/sigan"+".ckpt")
                return True;
            except:
                return False
        # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        #     self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        #     print(" [*] Success to read {}".format(ckpt_name))
        #     self.counter = int(ckpt_name.rsplit('-', 1)[-1])
        #     print(" [*] Start counter from {}".format(self.counter))
        #     return True
        # else:
        #     print(" [*] Failed to find a checkpoint under {}".format(checkpoint_dir))
        #     return False

    def dc_run(self):
        sound_train, image_train = self.load_data()
        n_classes = 44
        batch_size = 30
        n_images_train = image_train.shape[0]*image_train.shape[1]
        # n_batch_epoch = int(n_images_train / batch_size)
        max_iterations = 100 #1e5
        n_epoch = 50
        print_freq = 1
        sample_size = batch_size
        z_dim = 512
        sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
        alpha = 0.2
        image_size = 64
        layer = 18
        lr = 0.0002
        lr_decay = 0.5      
        decay_every = 100
        beta1 = 0.5


        #Critic
        critic_iterations = 5
        clip_values = (-0.01, 0.01)

        #Summary
        # tf.histogram_summary("z", self.z_vec)
        # tf.image_summary("image_real", self.images, max_images=2)
        # tf.image_summary("image_generated", self.gen_images, max_images=2)

        # Checkpoint Creation
        tl.files.exists_or_mkdir("samples/step1_gan-cls")
        tl.files.exists_or_mkdir("samples/step_pretrain_encoder")
        tl.files.exists_or_mkdir("checkpoint")
        save_dir = "checkpoint"


        # Placeholders
        t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
        t_real_sound = tf.placeholder(dtype=tf.float32, shape=[None,220500,1,1], name='real_sound_input')
        t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
        t_wrong_sound = tf.placeholder(dtype=tf.float32, shape=[None, 220500,1,1], name='wrong_sound_input')
        t_z = tf.placeholder(tf.float32, [None, z_dim], name='z_noise')


        # CNN Encoder
        net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
        x = net_cnn.outputs
        x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs

        #Instance of generator and discriminator
        generator_sound2img = dc_model.generator_txt2img_resnet
        discriminator_sound2img = dc_model.discriminator_txt2img_resnet


        #Training Inference for Sound to Image
        net_snn = self.sound_encoder1.layers[layer]
        net_fake_image, _ = generator_sound2img(t_z,
                    net_snn,
                    is_train=True, reuse=False, batch_size=batch_size)

        # Testing Inference for Sound to Image
        net_g, _ = generator_sound2img(t_z,
                    net_snn,
                    is_train=False, reuse=True, batch_size=batch_size)

        # Set up discriminator
        net_d, disc_fake_image_logits = discriminator_sound2img(
                            net_fake_image.outputs, net_snn, is_train=True, reuse=False)

        _, disc_real_image_logits = discriminator_sound2img(
                            t_real_image, net_snn, is_train=True, reuse=True)
            
        w_snn = self.sound_encoder2.layers[layer]
        _, disc_mismatch_logits = discriminator_sound2img(
                            t_real_image,
                            w_snn,
                            is_train=True, reuse=True)

        
        # Set up losses
        d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits,  tf.zeros_like(disc_mismatch_logits), name='d2')
        d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
        d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
        g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')


        # Get Variables for back propogation
        cnn_vars = tl.layers.get_variables_with_name('cnn', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)


        # Set up optimizers
        # with tf.variable_scope('learning_rate'):
        #     lr_v = tf.Variable(lr, trainable=False)
        # d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
        # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )

        #Set up optimizer
        optimizer = "RMSProp"
        learning_rate = 2e-4
        optimizer_param = 0.5

        optim = self._get_optimizer(optimizer, learning_rate, optimizer_param)
        g_optim = self._gradient_minimize(g_loss, g_vars, optim)
        d_optim = self._gradient_minimize(d_loss, d_vars, optim)

        # Initialize Variables
        tl.layers.initialize_global_variables(self.sess)

        train_variables = tf.trainable_variables()

        for v in train_variables:
            # print (v.op.name)
            add_to_regularization_and_summary(var=v)

        clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
                                         var in d_vars]
        #Load checkpoint
        if self.load_from_ckpt(save_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        #Set up tensorboard summary
        dloss1_sum = tf.summary.scalar("Real Loss", d_loss1)
        dloss2_sum = tf.summary.scalar("Mismatch Loss", d_loss2)
        dloss3_sum = tf.summary.scalar("Fake Loss", d_loss3)
        dloss_sum = tf.summary.scalar("Discriminator Loss", d_loss)
        gloss_sum = tf.summary.scalar("Generator Loss", g_loss)

        # gan_sum = tf.summary.merge([dloss1_sum, dloss2_sum, dloss3_sum, dloss_sum])
        gan_sum = tf.summary.merge_all()
        tboard_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 0
        # for epoch in range(0, n_epoch+1):
        #     start_time = time.time()

            # if epoch !=0 and (epoch % decay_every == 0):
            #     new_lr_decay = lr_decay ** (epoch // decay_every)
            #     self.sess.run(tf.assign(lr_v, lr * new_lr_decay))
            #     log = " ** new learning rate: %f" % (lr * new_lr_decay)
            #     # print(log)
            #     # logging.debug(log)
            # elif epoch == 0:
            #     log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            #     # print(log)

        start_time = time.time()
        for step in range(max_iterations):
            step_time = time.time()

            if step < 25 or step % 500 == 0:
                critic_itrs = 25
            else:
                critic_itrs = critic_iterations

            for critic_itr in range(critic_itrs):
                
                ind = random.sample(range(n_classes), int(n_classes/2))
                ind_ = np.setdiff1d(np.arange(n_classes),ind)

                idexs_w = get_random_int(min=0, max=image_train.shape[1]*len(ind)-1, number=batch_size)

                mis_sound = sound_train[ind].reshape(-1,sound_train.shape[2],sound_train.shape[3],sound_train.shape[4])
                mis_image = image_train[ind_].reshape(-1,image_train.shape[2],image_train.shape[3],image_train.shape[4])
                print(mis_sound.shape,mis_image.shape,idexs_w)
                wrong_sound = mis_sound[idexs_w]
                wrong_images = mis_image[idexs_w]

                
                # Today
                del(mis_sound)
                del(mis_image)
                sound_train_flat = sound_train.reshape(-1,sound_train.shape[2],sound_train.shape[3],sound_train.shape[4])
                image_train_flat = image_train.reshape(-1,image_train.shape[2],image_train.shape[3],image_train.shape[4])
                #########################################################
                
                idexs_r = get_random_int(min=0, max=image_train.shape[1]*image_train.shape[0]-1, number=batch_size)

                real_images = image_train_flat[idexs_r]
                real_sound = sound_train_flat[idexs_r]

                # Test sample
                idexs_t = get_random_int(min=0, max=image_train.shape[1]*image_train.shape[0]-1, number=30)
                test_sound = sound_train_flat[idexs_t]
                test_images = image_train_flat[idexs_t]

                del(sound_train_flat)
                del(image_train_flat)
                
                #del sound_train,image_train,image_train_flat,sound_train_flat
                ## get noise
                b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
                b_real_images = threading_data(real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
                b_wrong_images = threading_data(wrong_images, prepro_img, mode='train')
                print(b_real_images.shape)
                # Discriminator optimize
                print("Dis critic started")
                self.sess.run([d_optim], 
                                feed_dict={
                                            t_real_image : b_real_images,
                                            self.sound_encoder2.sound_input_placeholder: wrong_sound,
                                            self.sound_encoder1.sound_input_placeholder: real_sound,
                                            t_z : b_z})
                # Clip Optimize
                self.sess.run(clip_discriminator_var_op)
                print("Dis critic completed")
            if step % 200 == 0:
                counter += 1
                g_loss_val, d_loss_val, gan_sum_string = self.sess.run([g_loss, d_loss, gan_sum],feed_dict={
                                        t_real_image : b_real_images,
                                        self.sound_encoder2.sound_input_placeholder: wrong_sound,
                                        self.sound_encoder1.sound_input_placeholder: real_sound,
                                        t_z : b_z})
                print('-----------------------------------------------')
                print(step)
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (
                    step, g_loss_val, d_loss_val))

            ## updates G
                # errG, _, gloss_sum_string = self.sess.run([g_loss, g_optim, gloss_sum], feed_dict={
                #             self.sound_encoder1.sound_input_placeholder : real_sound,
                #             t_z : b_z})

                # tboard_writer.add_summary(gloss_sum_string, counter)
                tboard_writer.add_summary(gan_sum_string, counter)

            # print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.4f, g_loss: %.4f" \
            #             % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG))
            

            if(step%50 == 0):
                ni = int(np.ceil(np.sqrt(batch_size)))

                img_gen, snn_out = self.sess.run([net_g.outputs, net_snn], feed_dict={
                                                        self.sound_encoder1.sound_input_placeholder: test_sound,
                                                        t_z : sample_seed})
                print("End of Run")
                save_images(img_gen, [ni, ni], 'samples/step1_gan-cls/train_{:02d}.png'.format(step))
                
                save_images(test_images, [ni, ni], 'samples/real/train_{:02d}.png'.format(step))

            if(step!=0 and (step % 5000)==0):
                model_path = self.saver.save(self.sess, save_dir+"/sigan"+str(step)+".ckpt")

            ## save model
            # if (epoch != 0) and (epoch % 5) == 0:
            #     tl.files.save_npz(net_cnn.all_params, name='net_cnn_name', sess=self.sess)
            #     tl.files.save_npz(net_g.all_params, name='net_g_name', sess=self.sess)
            #     tl.files.save_npz(net_d.all_params, name='net_d_name', sess=self.sess)
            #     print("[*] Save checkpoints SUCCESS!")

            # if (epoch != 0) and (epoch % 10) == 0:
            #     tl.files.save_npz(net_cnn.all_params, name='net_cnn_name'+str(epoch), sess=self.sess)
            #     tl.files.save_npz(net_g.all_params, name='net_g_name'+str(epoch), sess=self.sess)
            #     tl.files.save_npz(net_d.all_params, name='net_d_name'+str(epoch), sess=self.sess) 

        model_path = self.saver.save(self.sess, save_dir+"/sigan"+".ckpt")

def main():
    dg1 = WGAN()
    dg1.dc_run()

if __name__ == '__main__':
    main()

