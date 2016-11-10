from __future__ import division
import os
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
                 batch_size=1, sample_length=1024,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64, run_g=2,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='default', data_dir=None,
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
        #self.sample_size = sample_size
        #self.output_length = output_length
        self.sample_length = sample_length
        self.output_length = sample_length
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.run_g = run_g
        self.c_dim = c_dim

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
#        self.g_bn5 = batch_norm(name='g_bn5')
#        self.g_bn6 = batch_norm(name='g_bn6')
#        self.g_bn7 = batch_norm(name='g_bn7')

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        if audio_params:
            with open(audio_params, 'r') as f:
                self.audio_params = json.load(f)
            self.gf_dim = self.audio_params['gf_dim']
            self.df_dim = self.audio_params['df_dim']
            self.gfc_dim = self.audio_params['gfc_dim']
            self.dfc_dim = self.audio_params['dfc_dim']
        else:
            self.gf_dim = gf_dim
            self.df_dim = df_dim
            self.gfc_dim = gfc_dim
            self.dfc_dim = dfc_dim


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
            audio_batch = self.reader.dequeue(self.batch_size)
            #import IPython; IPython.embed()

        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)

        #G deprecated, this only applies for mnist
        # if self.y_dim:
        #     self.G = self.generator(self.z, self.y)
        #     self.D, self.D_logits = self.discriminator(self.images, self.y, reuse=False)

        #     self.sampler = self.sampler(self.z, self.y)
        #     self.D_, self.D_logits = self.discriminator(self.G, self.y, reuse=True)
        # else:
        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(audio_batch, include_fourier=self.use_fourier)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True, include_fourier=self.use_fourier)

        
        #import IPython; IPython.embed()
        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        #G need to check sample rate
        self.G_sum = tf.audio_summary("G", self.G, sample_rate=self.audio_params['sample_rate'])

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        #G experiments with losses
        #self.max_like_g_loss = tf.reduce_mean(-tf.exp(self.D_logits_)/2.)
        #self.g_loss = self.max_like_g_loss
        #import IPython; IPython.embed()
        #self.g_loss = self.g_loss - self.d_loss

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

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



    def train(self, config):
        """Train DCGAN"""
        #G
        # if config.dataset == 'wav':
        #     coord = tf.train.Coordinator()
        #     reader = self.load_wav(coord)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        #import IPython; IPython.embed()
        init = tf.initialize_all_variables()
        self.sess.run(init)


        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        
        self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        counter = 1
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
                if config.dataset == 'wav':
                    batch_idxs = min(num_per_epoch, self.reader.corpus_size) // config.batch_size

                for idx in range(0, batch_idxs):
                    #G
                    if config.dataset == 'wav':
                        pass

                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
                    #G
                    if config.dataset == 'wav':
                        # audio_batch = reader.dequeue(self.batch_size) 
                        # audio_batch = audio_batch.eval()
                        # Update D network
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                            feed_dict={ self.z: batch_z })
                        self.writer.add_summary(summary_str, counter)

                        # Update G network run_g times
                        for i in range(self.run_g):
                            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                feed_dict={ self.z: batch_z })
                            self.writer.add_summary(summary_str, counter)

                        # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        # _, summary_str = self.sess.run([g_optim, self.g_sum],
                        #     feed_dict={ self.z: batch_z })
                        # self.writer.add_summary(summary_str, counter)


                        errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                        #errD_real = self.d_loss_real.eval({self.audio_samples: audio_batch})
                        errD_real = self.d_loss_real.eval()
                        errG = self.g_loss.eval({self.z: batch_z})
                        #G average over batch
                        #D_real = self.D.eval({self.audio_samples: audio_batch}).mean()
                        D_real = self.D.eval().mean()
                        D_fake = self.D_.eval({self.z: batch_z}).mean()


                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, D: %.3f, D_: %.3f" \
                        % (epoch+1, idx+1, batch_idxs,
                            time.time() - start_time, errD_fake+errD_real, errG, D_real, D_fake))

                    if np.mod(counter, config.save_every) == 1:
                        #G
                        if config.dataset == 'wav':
                            # samples, d_loss, g_loss = self.sess.run(
                            #     [self.sampler, self.d_loss, self.g_loss],
                            #     feed_dict={self.z: sample_z, self.images: sample_images.eval()}
                            # )
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={self.z: batch_z}
                            )
                            #import IPython; IPython.embed()
                        # Saving samples
                        if config.dataset == 'wav':
                            im_title = "d_loss: %.5f, g_loss: %.5f" % (d_loss, g_loss)
                            file_str = '{:02d}_{:04d}'.format(epoch, idx)
                            save_waveform(samples,config.out_dir+'/samples/train_'+file_str, title=im_title)
                            im_sum = get_im_summary(samples, title=file_str+im_title)
                            summary_str = self.sess.run(im_sum)
                            self.writer.add_summary(summary_str, counter)
                            
                            save_audios(samples[0], config.out_dir+'/samples/train_'+file_str+'.wav', 
                                format='.wav', sample_rate=self.audio_params['sample_rate'])
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    if np.mod(counter, 500) == 2:
                        self.save(config.out_dir+'/checkpoint', counter)
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

    def discriminator(self, audio_sample, y=None, reuse=False, include_fourier=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h4_dim = self.sample_length*self.df_dim//32

        h0 = lrelu(conv1d(audio_sample, self.df_dim, name='d_h0_conv'))
#        h1 = conv_bn_lrelu_layer(h0, self.df_dim*2, self.d_bn1, name='d_h1_conv')
        h1 = lrelu(self.d_bn1(conv1d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv1d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv1d(h2, self.df_dim*8, name='d_h3_conv')))
        #import IPython; IPython.embed()
        if self.use_disc: #not yet supported
            h_disc = mb_disc_layer(tf.reshape(h3, [self.batch_size, -1]),name='mb_disc')
            h4 = linear(h_disc, 1, name='d_h3_lin')
        else:
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, name='d_h3_lin',missing_dim=h4_dim)

        if include_fourier:
            fourier_sample = get_fourier(audio_sample)
            h0_f = lrelu(conv1d(fourier_sample, self.df_dim, name='d_h0_f_conv'))
            h1_f = lrelu(self.d_bn1f(conv1d(h0_f, self.df_dim*2, name='d_h1_f_conv')))
            h2_f = lrelu(self.d_bn2f(conv1d(h1_f, self.df_dim*4, name='d_h2_f_conv')))
            h3_f = lrelu(self.d_bn3f(conv1d(h2_f, self.df_dim*8, name='d_h3_f_conv')))
            #import IPython; IPython.embed()
            if self.use_disc: #not yet supported
                h_f_disc = mb_disc_layer(tf.reshape(h3_f, [self.batch_size, -1]),name='f_mb_disc')
                h4_f = linear(h_f_disc, 1, name='d_h3_f_lin')
            else:
                h4_f = linear(tf.reshape(h3_f, [self.batch_size, -1]), 1, name='d_h3_f_lin', missing_dim=h4_dim)
            h5 = linear(tf.concat(1,[h4,h4_f]),1, name='d_h5')
            #h5 = (h4+h4_f)/2
        else:
            h5 = h4
            #import IPython; IPython.embed()

        return tf.nn.sigmoid(h5), h5

    def generator(self, z, y=None):

        s = self.output_length

        sh = [s//2, s//4, s//8, s//16]

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.batch_size*sh[-1]*self.c_dim, 'g_h0_lin', with_w=True)

        h0 = tf.nn.relu(self.g_bn0(self.z_))

        self.h1, self.h1_w, self.h1_b = linear(h0, self.batch_size*sh[-2]*self.c_dim, 'g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        self.h2, self.h2_w, self.h2_b = linear(h1, self.batch_size*sh[-3]*self.c_dim, 'g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(self.h2))

        self.h3, self.h3_w, self.h3_b = linear(h2, self.batch_size*sh[-4]*self.c_dim, 'gh_3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(self.h3))

        self.h4, self.h4_w, self.h4_b = linear(h3, self.batch_size*s*self.c_dim, 'gh_4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(self.h4))

        h5 = tf.reshape(h4, [self.batch_size, s, self.c_dim])
        #import IPython; IPython.embed()

        return tf.nn.tanh(h5)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        s = self.output_length

        sh = [s//2, s//4, s//8, s//16]

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.batch_size*sh[-1]*self.c_dim, 'g_h0_lin', with_w=True)

        h0 = tf.nn.relu(self.g_bn0(self.z_))

        self.h1, self.h1_w, self.h1_b = linear(h0, self.batch_size*sh[-2]*self.c_dim, 'g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        self.h2, self.h2_w, self.h2_b = linear(h1, self.batch_size*sh[-3]*self.c_dim, 'g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(self.h2))

        self.h3, self.h3_w, self.h3_b = linear(h2, self.batch_size*sh[-4]*self.c_dim, 'gh_3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(self.h3))

        self.h4, self.h4_w, self.h4_b = linear(h3, self.batch_size*s*self.c_dim, 'gh_4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(self.h4))

        h5 = tf.reshape(h4, [self.batch_size, s, self.c_dim])
        #import IPython; IPython.embed()

        return tf.nn.tanh(h5)


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
