from __future__ import division
import os, sys
import time
from glob import glob
import tensorflow as tf
import numpy as np
import json
from audio_reader import AudioReader
from ops import *
from utils import *
from postprocess import *

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=1, sample_length=1024, net_size_g = 512, net_depth_g = 6, net_size_q = 1024, keep_prob = 1.0,
                 y_dim=None, z_dim=100, df_dim=32, gf_dim=32, run_g=2, run_v=2, 
                 c_dim=1, dataset_name='default', model_name = "cppnvae", data_dir=None, 
                 audio_params=None, checkpoint_dir=None, out_dir=None, use_disc=False, use_fourier=True, mode='generate'):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            sample_length: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.net_size_g = net_size_g
        self.net_size_q = net_size_q
        self.sample_length = sample_length
        self.output_length = sample_length
        self.x_dim = sample_length
        #self.scale = scale
        self.net_depth_g = net_depth_g
        self.keep_prob = keep_prob
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.run_v = run_v
        self.run_g = run_g
        self.c_dim = c_dim
        self.model_name = model_name
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        if True:
            self.d_bn1f = batch_norm(name='d_bn1f')
            self.d_bn2f = batch_norm(name='d_bn2f')
            self.d_bn3f = batch_norm(name='d_bn3f')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn6 = batch_norm(name='g_bn6')
        self.g_bn7 = batch_norm(name='g_bn7')

        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        if audio_params:
            self.audio_params = audio_params
            self.df_dim = self.audio_params['df_dim']
            self.gf_dim = self.audio_params['gf_dim']
            self.learning_rate = self.audio_params['learning_rate']
            self.learning_rate_vae = self.audio_params['learning_rate_vae']
            self.learning_rate_d = self.audio_params['learning_rate_d']
            self.beta1 = self.audio_params['beta1']
            self.scale = self.sample_length/self.audio_params['sample_rate']

        else:
            self.df_dim = df_dim


        self.checkpoint_dir = checkpoint_dir
        self.use_disc = use_disc
        self.use_fourier = use_fourier
        if mode == 'train': self.build_model(self.dataset_name)

    def build_model(self, dataset):
        # if self.y_dim:
        #     self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        #G
        if dataset == 'wav':
            # self.audio_samples = tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
            #                         name='real_samples')
#            self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
#                                    name='real_images')
            # self.gen_audio_samples= tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
            #                             name='gen_audio_samples')
            self.coord = tf.train.Coordinator()
            self.reader = self.load_wav(self.coord)
            self.audio_batch = self.reader.dequeue(self.batch_size)
            self.batch_flatten = tf.reshape(self.audio_batch, [self.batch_size, -1])
            #import IPython; IPython.embed()
        self.x_vec = self.coordinates(self.x_dim, self.scale)
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, None, 1))

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = self.encoder()

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z_sum = tf.histogram_summary("z", self.z)

        #G deprecated, this only applies for mnist
        # if self.y_dim:
        #     self.G = self.generator(self.z, self.y)
        #     self.D, self.D_logits = self.discriminator(self.images, self.y, reuse=False)

        #     self.sampler = self.sampler(self.z, self.y)
        #     self.D_, self.D_logits = self.discriminator(self.G, self.y, reuse=True)
        # else:
        self.G = self.generator(gen_x_dim=self.sample_length)
        self.batch_reconstruct_flatten = tf.reshape(self.G, [self.batch_size, -1])
        self.D, self.D_logits = self.discriminator(self.audio_batch, include_fourier=self.use_fourier)
        #self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True, include_fourier=self.use_fourier)
        
        #import IPython; IPython.embed()
        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.audio_summary("G", self.G, sample_rate=self.audio_params['sample_rate'])

        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake

        #G experiments with losses
        #self.max_like_g_loss = tf.reduce_mean(-tf.exp(self.D_logits_)/2.)
        #self.g_loss = self.max_like_g_loss
        #import IPython; IPython.embed()
        #self.g_loss = self.g_loss - self.d_loss


        self.create_vae_loss_terms()
        self.create_gan_loss_terms()
        self.balanced_loss = 1.0 * self.g_loss + 1.0 * self.reconstr_loss # can try to weight these.

        self.t_vars = tf.trainable_variables()

        self.q_vars = [var for var in self.t_vars if 'q_' in var.name]
        self.g_vars = [var for var in self.t_vars if 'g_' in var.name]
        self.d_vars = [var for var in self.t_vars if 'd_' in var.name]
        self.vae_vars = self.q_vars #+self.g_vars

        # Use ADAM optimizer
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                          .minimize(self.balanced_loss, var_list=self.vae_vars)
        self.vae_optim = tf.train.AdamOptimizer(self.learning_rate_vae, beta1=self.beta1) \
                          .minimize(self.vae_loss, var_list=self.vae_vars)
        self.saver = tf.train.Saver()

    def create_vae_loss_terms(self):
        #Bournoulli MLP, for positive data
        #self.reconstr_loss = \
        #    -tf.reduce_sum(self.batch_flatten * tf.log(1e-10 + self.batch_reconstruct_flatten)
        #                   + (1-self.batch_flatten) * tf.log(1e-10 + 1 - self.batch_reconstruct_flatten), 1)
        #Gaussian MLP
        self.reconstr_loss = 0.5*tf.reduce_mean(tf.square(self.batch_flatten-self.batch_reconstruct_flatten)/1)
        # L2
        #self.reconstr_loss = tf.reduce_mean(tf.square(self.batch_flatten-self.batch_reconstruct_flatten))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.vae_loss = tf.reduce_mean(self.reconstr_loss + self.latent_loss) / self.x_dim 
        # average over batch and pixel
        #import IPython; IPython.embed()
        self.r_loss_sum = tf.scalar_summary("r_loss", self.reconstr_loss)
        self.l_loss_sum = tf.scalar_summary(["l_loss"], self.latent_loss)
        self.vae_loss_sum = tf.scalar_summary("vae_loss", self.vae_loss)

    def create_gan_loss_terms(self):
        # Define loss function and optimiser
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/2.
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

    def coordinates(self, x_dim = 1024, scale = 1.0):
        x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        x_vec = np.tile(x_range.flatten(), self.batch_size).reshape(self.batch_size, x_dim, 1)
        #import IPython; IPython.embed()
        return x_vec

    
    def encoder(self):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        H1 = tf.nn.dropout(tf.nn.softplus(linear(self.batch_flatten, self.net_size_q, self.model_name+'_q_lin1', \
                     missing_dim=self.sample_length)), self.keep_prob)
        H2 = tf.nn.dropout(tf.nn.softplus(linear(H1, self.net_size_q, self.model_name+'_q_lin2')), self.keep_prob)
        z_mean = linear(H2, self.z_dim, self.model_name+'_q_lin3_mean')
        z_log_sigma_sq = linear(H2, self.z_dim, self.model_name+'_q_lin3_log_sigma_sq')
        return (z_mean, z_log_sigma_sq)

    def discriminator(self, audio_sample, y=None, reuse=False, include_fourier=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h4_dim = self.sample_length*self.df_dim//32

        h0 = lrelu(conv1d(audio_sample, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv1d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv1d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv1d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, name='d_h3_lin',missing_dim=h4_dim)

        if include_fourier:
            fourier_sample = get_fourier(audio_sample)
            h0_f = lrelu(conv1d(fourier_sample, self.df_dim, name='d_h0_f_conv'))
            h1_f = lrelu(self.d_bn1f(conv1d(h0_f, self.df_dim*2, name='d_h1_f_conv')))
            h2_f = lrelu(self.d_bn2f(conv1d(h1_f, self.df_dim*4, name='d_h2_f_conv')))
            h3_f = lrelu(self.d_bn3f(conv1d(h2_f, self.df_dim*8, name='d_h3_f_conv')))
            h4_f = linear(tf.reshape(h3_f, [self.batch_size, -1]), 1, name='d_h3_f_lin', missing_dim=h4_dim)
            h5 = linear(tf.concat(1,[h4,h4_f]),1, name='d_h5')
            #h5 = (h4+h4_f)/2
        else:
            h5 = h4
            #import IPython; IPython.embed()

        return tf.nn.sigmoid(h5), h5

    # def generator(self, gen_x_dim = 1024, reuse = False):

    #     if reuse:
    #         tf.get_variable_scope().reuse_variables()

    #     n_network = self.net_size_g
    #     z_scaled = tf.reshape(self.z, [self.batch_size, 1, self.z_dim]) * \
    #                     tf.ones([gen_x_dim, 1], dtype=tf.float32) * self.scale
    #     z_unroll = tf.reshape(z_scaled, [self.batch_size*gen_x_dim, self.z_dim])
    #     x_unroll = tf.reshape(self.x, [self.batch_size*gen_x_dim, 1])

    #     U = fully_connected(z_unroll, n_network, self.model_name+'_g_0_z') + \
    #         fully_connected(x_unroll, n_network, self.model_name+'_g_0_x', with_bias = False)
    #     H = tf.nn.softplus(U)
    #     for i in range(1, self.net_depth_g):
    #         H = tf.nn.tanh(fully_connected(H, n_network, self.model_name+'_g_tanh_'+str(i)))

    #     output = tf.sigmoid(fully_connected(H, self.c_dim, self.model_name+'_g_'+str(self.net_depth_g)))
    #     result = tf.reshape(output, [self.batch_size, gen_x_dim, self.c_dim])

    #     return result

    def generator(self, gen_x_dim = 1024, reuse = False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        s = gen_x_dim

        sh = [s//2, s//4, s//8, s//16, s//32, int(s/32/4**1), int(s/32/4**2), int(s/32/4**3)]

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(self.z, self.gf_dim*128*sh[-1], 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, sh[-1], self.gf_dim * 128])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv1d(h0, 
            [self.batch_size, sh[-2], self.gf_dim*64], d_w=4, name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv1d(h1,
            [self.batch_size, sh[-3], self.gf_dim*32], d_w=4, name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv1d(h2,
            [self.batch_size, sh[-4], self.gf_dim*16], d_w=4, name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv1d(h3,
            [self.batch_size, sh[-5], self.gf_dim*8], d_w=2, name='g_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4))
        h5, self.h5_w, self.h5_b = deconv1d(h4,
            [self.batch_size, sh[-6], self.gf_dim*4], d_w=2, name='g_h5', with_w=True)
        h5 = tf.nn.relu(self.g_bn5(h5))
        h6, self.h6_w, self.h6_b = deconv1d(h5,
            [self.batch_size, sh[-7], self.gf_dim*2], d_w=2, name='g_h6', with_w=True)
        h6 = tf.nn.relu(self.g_bn6(h6))
        h7, self.h7_w, self.h7_b = deconv1d(h6,
            [self.batch_size, sh[-8], self.gf_dim*1], d_w=2, name='g_h7', with_w=True)
        h7 = tf.nn.relu(self.g_bn7(h7))
        h8, self.h8_w, self.h8_b = deconv1d(h7,
            [self.batch_size, s, self.c_dim], d_w=2, name='g_h8', with_w=True)

        return tf.nn.tanh(h8)

    def encode(self, X):
      """Transform data by mapping it into the latent space."""
      # Note: This maps to mean of distribution, we could alternatively
      # sample from Gaussian distribution
      return self.sess.run(self.z_mean, feed_dict={self.batch: X})

    def sampler(self, gen_x_dim = 1024):
        tf.get_variable_scope().reuse_variables()

        s = gen_x_dim
        sh = [s//2, s//4, s//8, s//16, s//32, int(s/32/4**1), int(s/32/4**2), int(s/32/4**3)]

        # project `z` and reshape
        self.z_ = linear(self.z, self.gf_dim*128*sh[-1], 'g_h0_lin')

        self.h0 = tf.reshape(self.z_, [-1, sh[-1], self.gf_dim * 128])
        h0 = tf.nn.relu(self.g_bn0(self.h0, train=False))

        self.h1 = deconv1d(h0, 
            [self.batch_size, sh[-2], self.gf_dim*64], d_w=4, name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(self.h1, train=False))

        h2 = deconv1d(h1,
            [self.batch_size, sh[-3], self.gf_dim*32], d_w=4, name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv1d(h2,
            [self.batch_size, sh[-4], self.gf_dim*16], d_w=4, name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv1d(h3,
            [self.batch_size, sh[-5], self.gf_dim*8], d_w=2, name='g_h4')
        h4 = tf.nn.relu(self.g_bn4(h4, train=False))

        h5 = deconv1d(h4,
            [self.batch_size, sh[-6], self.gf_dim*4], d_w=2, name='g_h5')
        h5 = tf.nn.relu(self.g_bn5(h5, train=False))
        h6 = deconv1d(h5,
            [self.batch_size, sh[-7], self.gf_dim*2], d_w=2, name='g_h6')
        h6 = tf.nn.relu(self.g_bn6(h6, train=False))

        h7 = deconv1d(h6,
            [self.batch_size, sh[-8], self.gf_dim*1], d_w=2, name='g_h7')
        h7 = tf.nn.relu(self.g_bn7(h7))
        h8 = deconv1d(h7,
            [self.batch_size, s, self.c_dim], d_w=2, name='g_h8')
        return tf.nn.tanh(h8)

    def generate(self, config):
        '''generate samples from trained model'''
        self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        for counter in range(config.gen_size):
            batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                    .astype(np.float32)
            samples = self.sess.run(self.sampler, feed_dict={self.z: batch_z})
            file_str = '{:03d}'.format(counter)

            #samples = pc_chop(samples,100) #postprocess
            #samples = (samples*32767.0).astype('int16')
            #import IPython; IPython.embed()
            #wav_sum_str = tf.audio_summary('S',samples,sample_rate=self.audio_params['sample_rate'],max_outputs=10)
            #self.writer.add_summary(wav_sum_str, counter)
            #import IPython; IPython.embed()

            save_waveform(samples,config.out_dir+'/'+file_str, title='')
            im_sum = get_im_summary(samples, title=file_str)
            summary_str = self.sess.run(im_sum)
            self.writer.add_summary(summary_str, counter)
            
            save_audios(samples[0], config.out_dir+'/'+file_str+'.wav', 
                format='.wav', sample_rate=self.audio_params['sample_rate'])


    def make_step(self):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.

        I should really seperate the below tricks into parameters, like number of times/pass
        and also the regulator threshold levels.
        """

        op_count = 0
        '''
        for i in range(4):
          counter += 1
          _, vae_loss, g_loss = self.sess.run((self.g_opt, self.vae_loss, self.g_loss),
                                  feed_dict={self.batch: batch, self.x: self.x_vec, self.y: self.y_vec, self.r: self.r_vec})
          if g_loss < 0.6:
            break
        '''

        for i in range(self.run_v):
          #op_count += 1
          _, vae_loss = self.sess.run((self.vae_optim, self.vae_loss), feed_dict={self.x: self.x_vec})

        for i in range(self.run_g):
          #op_count += 1
          _, g_loss = self.sess.run((self.g_optim, self.g_loss), feed_dict={self.x: self.x_vec})
          if g_loss < 0.6: break

        d_loss = self.d_loss.eval({self.x: self.x_vec})
        if d_loss > 0.45 and g_loss < 0.85:
          for i in range(1):
            op_count += 1
            _, d_loss = self.sess.run((self.d_optim, self.d_loss), feed_dict={self.x: self.x_vec})
            if d_loss < 0.6: break
        v_sumstr, g_sumstr, d_sumstr = self.sess.run((self.v_sum, self.g_sum, self.d_sum), feed_dict={self.x: self.x_vec})
        self.writer.add_summary(v_sumstr, self.counter)
        self.writer.add_summary(g_sumstr, self.counter)
        self.writer.add_summary(d_sumstr, self.counter)

        return d_loss, g_loss, vae_loss, op_count

    def train(self, config):
        """Train DCGAN"""
        #G
        # if config.dataset == 'wav':
        #     coord = tf.train.Coordinator()
        #     reader = self.load_wav(coord)

        init = tf.initialize_all_variables()
        self.sess.run(init)
        #import IPython; IPython.embed()


        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.v_sum = tf.merge_summary([self.vae_loss_sum, self.r_loss_sum, self.l_loss_sum])
        self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        # sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        self.counter = 1
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Successfully Loaded Checkpoint")
        else:
            print(" [!] Unable to load checkpoint. Skipping...")

        #G
        if config.dataset == 'wav':
            threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.reader.start_threads(self.sess)
            num_per_epoch = self.audio_params['num_per_epoch'] 
        #G
        try:
            for epoch in range(config.epoch):
                avg_d_loss = 0.
                avg_q_loss = 0.
                avg_vae_loss = 0.
                if config.dataset == 'wav':
                    batch_idxs = min(num_per_epoch, self.reader.corpus_size) // config.batch_size

                for idx in range(0, batch_idxs):

                    #G
                    if config.dataset == 'wav':


                        # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        # _, summary_str = self.sess.run([g_optim, self.g_sum],
                        #     feed_dict={ self.z: batch_z })
                        # self.writer.add_summary(summary_str, counter)
                        errD, errG, errV, n_operations = self.make_step()

                        D_real = self.D.eval().mean()
                        D_fake = self.D_.eval({self.x: self.x_vec}).mean()


                    self.counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.5f, g_loss: %.5f, v_loss: %.5f, D: %.3f, D_: %.3f, n_ops: %1d" \
                        % (epoch+1, idx+1, batch_idxs,
                            time.time() - start_time, errD, errG, errV, D_real, D_fake, n_operations))
                    if np.mod(self.counter, config.print_every) == 1: sys.stdout.flush()

                    if np.mod(self.counter, config.save_every) == 1:
                        #G
                        if config.dataset == 'wav':
                            # samples, d_loss, g_loss = self.sess.run(
                            #     [self.sampler, self.d_loss, self.g_loss],
                            #     feed_dict={self.z: sample_z, self.images: sample_images.eval()}
                            # )
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler(gen_x_dim=self.x_dim), self.d_loss, self.g_loss],
                                feed_dict={self.x: self.x_vec}
                            )
                            #import IPython; IPython.embed()
                        # Saving samples
                        if config.dataset == 'wav':
                            im_title = "d_loss: %.5f, g_loss: %.5f" % (d_loss, g_loss)
                            file_str = '{:02d}_{:04d}'.format(epoch, idx)
                            save_waveform(samples,config.out_dir+'/samples/train_'+file_str, title=im_title)
                            im_sum = get_im_summary(samples, title=file_str+im_title)
                            summary_str = self.sess.run(im_sum)
                            self.writer.add_summary(summary_str, self.counter)
                            
                            save_audios(samples[0], config.out_dir+'/samples/train_'+file_str+'.wav', 
                                format='.wav', sample_rate=self.audio_params['sample_rate'])
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(self.counter, 500) == 2:
                        self.save(config.out_dir+'/checkpoint', self.counter)
        #G
        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        #G
        finally:
            if config.dataset == 'wav':
                print('Finished, output see {}'.format(config.out_dir))
                self.coord.request_stop()
                self.coord.join(threads)
    #G
    def load_wav(self, coord):
        if self.data_dir:
            data_dir = self.data_dir
        else:
            data_dir = os.path.join("./data", self.dataset_name)
        EPSILON = 0.001
        silence_threshold = self.audio_params['silence_threshold'] if self.audio_params['silence_threshold'] > \
                                                      EPSILON else None
        reader = AudioReader(
            data_dir,
            coord,
            sample_rate=self.audio_params['sample_rate'],
            sample_length=self.sample_length,
            silence_threshold=silence_threshold)
        return reader


    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_length)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_length)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
