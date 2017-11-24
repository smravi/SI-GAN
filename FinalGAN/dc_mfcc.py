""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
import scipy.misc
from scipy.io import loadmat
import time, os, re
import librosa

from utils import *
from dc_model import *
import dc_model

# Set the path for Sound Encoder
# import sys
# sys.path.append('./Encoder/')

# Import Soundnet
# from Encoder import *

class WGAN:
    def __init__(self):
        self.sess           = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # self.param_G        = np.load('./models/sound5.npy', encoding='latin1').item()
        # self.sound_config ={
        #     'batch_size': 1,
        #     'train_size': np.inf,
        #     'epoch': 5,
        #     'eps': 1e-5,
        #     'learning_rate': 1e-3,
        #     'beta1': 0.9,
        #     'load_size': 22050*4,
        #     'sample_rate': 22050,
        #     'name_scope': 'SoundNet',
        #     'phase': 'train',
        #     'dataset_name': 'ESC50',
        #     'subname': 'mp3',
        #     'checkpoint_dir': 'checkpoint',
        #     'dump_dir': 'output',
        #     'model_dir': None,
        #     'param_g_dir': './models/sound5.npy',
        #     }
        #self.EncodeModel()


    # def EncodeModel(self):
    #     self.sound_config['name_scope'] = 'Soundnet1'
    #     self.sound_encoder1 = Model(self.sess, config=self.sound_config,param_G=self.param_G,fcparam_G=None)
    #     self.sound_config['name_scope'] = 'Soundnet2'
    #     self.sound_encoder2 = Model(self.sess, config=self.sound_config, param_G=self.param_G,fcparam_G=None)

    def load_data(self):
        datapath = './data/'
        # self.sound_train = np.load(datapath+'esc_44_sound.npy')
        # self.image_train = np.load(datapath+'esc_44_image.npy')
        self.sound_train = np.load(datapath+'cust_esc_10_sound.npy')
        self.image_train = np.load(datapath+'cust_esc_10_images.npy')
        self.key_labels = np.load(datapath+'cust_esc_10_keys.npy')
    
    def aug_samples(self):
        n_samples=40
        real_map = []
        for cat in range(self.n_classes):
            for clip in range(n_samples):
                for img in range(n_samples):
                    real_map.append((cat, clip, img))
        real_map = np.array(real_map)

        mis_map = []
        for cat in range(self.n_classes):
            for clip in range(n_samples):
                for img in range(n_samples):
                    rlist = []
                    for i in range(10) :
                        if i!=cat:
                            rlist.append(i)
                    cat_ = random.choice(rlist)
                    mis_map.append((cat, clip, cat_,img))
        mis_map = np.array(mis_map)
        return real_map, mis_map

    def get_batch(self, real=True):
        if real:
            idx = random.sample(range(self.real_map.shape[0]), self.batch_size)
            batch_images = self.image_train[self.real_map[idx][:,0],self.real_map[idx][:,2]]
            batch_sound = self.sound_train[self.real_map[idx][:,0],self.real_map[idx][:,1]]
            batch_labels = self.real_map[idx][:,0]
        else:
            idx = random.sample(range(self.mis_map.shape[0]), self.batch_size)
            batch_images = self.image_train[self.mis_map[idx][:,2],self.mis_map[idx][:,3]]
            batch_sound = self.sound_train[self.mis_map[idx][:,0],self.mis_map[idx][:,1]]
            batch_labels = self.mis_map[idx][:,0]
        return batch_sound, batch_images, batch_labels

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

        # path = checkpoint_dir+"/sigan"+".ckpt"
        # if(os.path.exists(checkpoint_dir+"/sigan"+".ckpt")):
        #     try:
        #         self.saver.restore(self.sess, checkpoint_dir+"/sigan"+".ckpt")
        #         return True;
        #     except:
        #         return False

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('Checkpoint Name:{}'.format(ckpt_name))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            self.counter = int(ckpt_name.rsplit('-', 1)[-1])
            print(" [*] Start counter from {}".format(self.counter))
            return True
        else:
            print(" [*] Failed to find a checkpoint under {}".format(checkpoint_dir))
            return False

    def dc_run(self):
        self.n_classes = 10
        self.batch_size = 30

        # Loading data from numpy file
        self.load_data()
        # Multiplexing data
        self.real_map, self.mis_map = self.aug_samples()

        n_batch_epoch = int(self.real_map.shape[0] / self.batch_size)
        # n_batch_epoch = 3
        n_epoch = 50
        print_freq = 1
        z_dim = 101
        sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, z_dim)).astype(np.float32)
        alpha = 0.2
        image_size = 64
        layer = 18
        lr = 0.0002
        lr_decay = 0.5      
        decay_every = 500000
        max_iterations = 20000 #1e5
        learning_rate = 2e-4
        optimizer_param = 0.5 #beta1 = 0.5
        ni = int(np.ceil(np.sqrt(self.batch_size)))
        #Set up optimizer
        optimizer = "Adam"

        # Checkpoint Creation
        tl.files.exists_or_mkdir("samples/step1_gan-cls")
        tl.files.exists_or_mkdir("samples/real")
        tl.files.exists_or_mkdir("samples/step_pretrain_encoder")
        tl.files.exists_or_mkdir("checkpoint")
        save_dir = "checkpoint"


        # Placeholders
        t_real_image = tf.placeholder('float32', [None, image_size, image_size, 3], name = 'real_image')
        t_real_sound = tf.placeholder(dtype=tf.float32, shape=[None,27], name='real_sound_input')
        t_wrong_image = tf.placeholder('float32', [None ,image_size, image_size, 3], name = 'wrong_image')
        t_wrong_sound = tf.placeholder(dtype=tf.float32, shape=[None, 27], name='wrong_sound_input')
        t_z = tf.placeholder(tf.float32, [None, z_dim], name='z_noise')
        t_z = tf.placeholder(tf.float32, [None, z_dim], name='z_noise')

        t_labels = tf.placeholder(tf.int64, [None])
        labels_hot = tf.one_hot(t_labels, self.n_classes)


        # CNN Encoder
        net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
        x = net_cnn.outputs
        x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs

        #Instance of generator and discriminator
        generator_sound2img = dc_model.generator_txt2img_resnet
        discriminator_sound2img = dc_model.discriminator_txt2img_resnet


        #Training Inference for Sound to Image
        net_snn = t_real_sound #self.sound_encoder1.layers[layer]
        net_fake_image, _ = generator_sound2img(t_z,
                    net_snn,
                    is_train=True, reuse=False, batch_size=self.batch_size)

        # Testing Inference for Sound to Image
        net_g, _ = generator_sound2img(t_z,
                    net_snn,
                    is_train=False, reuse=True, batch_size=self.batch_size)

        # Set up discriminator
        net_d, disc_fake_image_logits,net_feature_f = discriminator_sound2img(
                            net_fake_image.outputs, net_snn, is_train=True, reuse=False)

        _, disc_real_image_logits,net_feature_r = discriminator_sound2img(
                            t_real_image, net_snn, is_train=True, reuse=True)
            
        w_snn = t_wrong_sound   #self.sound_encoder2.layers[layer]
        _, disc_mismatch_logits,_ = discriminator_sound2img(
                            t_real_image,
                            w_snn,
                            is_train=True, reuse=True)

        
        # Set up losses
        d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits,  tf.zeros_like(disc_mismatch_logits), name='d2')
        d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
        
        # Categorical Loss
        labels_hot = tf.one_hot(t_labels, self.n_classes)
        # ce = tf.nn.softmax_cross_entropy_with_logits(logits=cat_logit_fake, labels=labels_hot)
        # fake_cat_loss = tf.reduce_mean(ce) 
        d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
        d_feature_loss = tf.reduce_mean(tf.nn.l2_loss(net_feature_f.outputs - net_feature_r.outputs, name='lf'))/tf.cast(tf.size(net_feature_f.outputs), dtype=tf.float32)
        g_loss_logit = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')
        g_loss = g_loss_logit + 0.2 * d_feature_loss


        # print('Labels shape:{}'.format(labels_hot))
        # # AC GAN
        # d_loss = d_loss + fake_cat_loss
        # g_loss = (g_loss + fake_cat_loss)

        # Get Variables for back propogation
        cnn_vars = tl.layers.get_variables_with_name('cnn', True, True)
        d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
        g_vars = tl.layers.get_variables_with_name('generator', True, True)


        # Learning Rate
        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr, trainable=False)
        # d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
        # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )



        optim = self._get_optimizer(optimizer, lr_v, optimizer_param)
        g_optim = self._gradient_minimize(g_loss, g_vars, optim)
        d_optim = self._gradient_minimize(d_loss, d_vars, optim)

        # Initialize Variables
        tl.layers.initialize_global_variables(self.sess)

        train_variables = tf.trainable_variables()

        for v in train_variables:
            # print (v.op.name)
            add_to_regularization_and_summary(var=v)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5,restore_sequentially=True)
        self.saver.restore(self.sess, "bkup_models/sigan.ckpt")
        #Load checkpoint
        # if self.load_from_ckpt(save_dir):
        #    print(" [*] Load SUCCESS")
        # else:
        #    print(" [!] Load failed...")
        #Set up tensorboard summary
        dloss1_sum = tf.summary.scalar("Real Loss", d_loss1)
        dloss2_sum = tf.summary.scalar("Mismatch Loss", d_loss2)
        dloss3_sum = tf.summary.scalar("Fake Loss", d_loss3)
        dloss_sum = tf.summary.scalar("Discriminator Loss", d_loss)
        gloss_sum = tf.summary.scalar("Generator Loss", g_loss)
        # cat_loss_sum = tf.summary.scalar("Classification Loss", fake_cat_loss)

        # gan_sum = tf.summary.merge([dloss1_sum, dloss2_sum, dloss3_sum, dloss_sum])

        # tf.histogram_summary("z", self.z_vec)
        # tf.image_summary("image_real", self.test_images, max_images=2)
        # tf.image_summary("image_generated", self.img_gen, max_images=2)

        gan_sum = tf.summary.merge_all()
        tboard_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        
        counter = 0
        print('Starting Training')
        start_time = time.time()
        for step in range(max_iterations):
            if step !=0 and (step % decay_every == 0):
                new_lr_decay = lr_decay ** (step // decay_every)
                self.sess.run(tf.assign(lr_v, lr * new_lr_decay))
                log = " ** new learning rate: %f" % (lr * new_lr_decay)
                    # print(log)
                    # logging.debug(log)
            elif step == 0:
                log = " ** init lr: %f  decay_every_step: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            step_time = time.time()
 
            real_sound, real_images,r_labels = self.get_batch(real = True)
            wrong_sound, wrong_images,w_labels = self.get_batch(real = False)

            # print(real_sound.shape)
            # print('r_labels:{}'.format(r_labels))
            # print('w_labels:{}'.format(w_labels))
            # print('Labels keys:{}'.format(self.key_labels))
            # print('Testing Data')
            # path = './test/'
            # for i in range(r_labels.size):
            #     print(r_labels[i])
            #     #clip = real_sound[i].reshape(-1)
            #     #librosa.output.write_wav(path+self.key_labels[r_labels[i]]+str(i)+'.wav', clip,sr=44100)
            #     # save_images(real_images, [ni, ni], 'test/train_{:02d}.png'.format(step))
            #     scipy.misc.imsave(path+self.key_labels[r_labels[i]]+str(i)+'.png', real_images[i])

            b_z = np.random.normal(loc=0.0, scale=1.0, size=(self.batch_size, z_dim)).astype(np.float32)
            b_real_images = threading_data(real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
            b_wrong_images = threading_data(wrong_images, prepro_img, mode='train')

            #if(step<200):
            # Discriminator optimize
            self.sess.run([d_optim], 
                                feed_dict={
                                            t_real_image : b_real_images,
                                            t_wrong_sound: wrong_sound,
                                            t_real_sound: real_sound,
                                            t_z : b_z})

            self.sess.run([g_optim], feed_dict={
                                t_real_image : b_real_images,
                                t_real_sound : real_sound,
                                t_z : b_z})

            
            # Loss Visualization
            counter += 1
            if(step!=0 and step % 100==0):
                g_loss_val, d_loss_val, gan_sum_string = self.sess.run([g_loss, d_loss, gan_sum],feed_dict={
                                        t_real_image : b_real_images,
                                        t_wrong_sound: wrong_sound,
                                        t_real_sound: real_sound,
                                        t_z : b_z})
                print('-----------------------------------------------')
                print("Step: %d, generator loss: %g, discriminator_loss: %g" % (step, g_loss_val, d_loss_val))
                # print('Classification Loss: %d' %(cat_loss))

                tboard_writer.add_summary(gan_sum_string, counter)
            

            if(step!=0 and step%100 == 0):
                self.test_sound, self.test_images,_ = self.get_batch(real = True)
                self.img_gen, self.snn_out = self.sess.run([net_g.outputs, net_snn], feed_dict={
                                                        t_real_sound : self.test_sound,
                                                        t_real_image: self.test_images,
                                                        t_z : sample_seed})
                print("Generating test image")
                save_images(self.img_gen, [ni, ni], 'samples/step1_gan-cls/train_{:02d}.png'.format(step))
                
                save_images(self.test_images, [ni, ni], 'samples/real/train_{:02d}.png'.format(step))

            
            if(step!=0 and (step % 500)==0):
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

        model_path = self.saver.save(self.sess, 'final_model'+"/sigan"+".ckpt")

def main():
    dg1 = WGAN()
    dg1.dc_run()

if __name__ == '__main__':
    main()

