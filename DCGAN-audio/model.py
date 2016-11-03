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
                 audio_params=None, checkpoint_dir=None, out_dir=None, use_disc=False, use_fourier=True,
                 num_d_layers=3, num_g_layers=3):
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
            num_d_layers: (optional) number of layers that use batch normalization in the discriminator [3]
            num_g_layers: (optional) number of layers that use batch normalization in the generator [3]
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
        self.num_d_layers = num_d_layers
        self.num_g_layers = num_g_layers

        # batch normalization : deals with poor initialization helps gradient flow
        self.dbn = []
        for n in range(1, self.num_d_layers+1):
            var_name = 'd_bn' + str(n)
            self.dbn.append(batch_norm(name=var_name))
#        self.d_bn1 = batch_norm(name='d_bn1')
#        self.d_bn2 = batch_norm(name='d_bn2')
#        self.d_bn3 = batch_norm(name='d_bn3')

        if use_fourier:
            self.dbnf = []
            for n in range(1, self.num_d_layers+1):
                var_name = 'd_bn' + str(n) + 'f'
                self.dbnf.append(batch_norm(name=var_name))
#            self.d_bn1f = batch_norm(name='d_bn1f')
#            self.d_bn2f = batch_norm(name='d_bn2f')
#            self.d_bn3f = batch_norm(name='d_bn3f')

        self.gbn = []
        for n in range(0, self.num_g_layers+1):
            var_name = 'g_bn' + str(n)
            self.gbn.append(batch_norm(name=var_name))
#        self.g_bn0 = batch_norm(name='g_bn0')
#        self.g_bn1 = batch_norm(name='g_bn1')
#        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.gbn.append(batch_norm(name='g_bn'+str(self.num_g_layers+1)))
#            self.g_bn3 = batch_norm(name='g_bn3')

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
        self.build_model(self.dataset_name)

    def build_model(self, dataset):
        # if self.y_dim:
        #     self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        #G
        if dataset == 'wav':
            self.audio_samples = tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
                                    name='real_samples')
#            self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
#                                    name='real_images')
            self.gen_audio_samples= tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
                                        name='gen_audio_samples')
#            self.sample_images= tf.placeholder(tf.float32, [self.batch_size] + [self.output_length, 1],
#                                        name='sample_images')
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
        self.D, self.D_logits = self.discriminator(self.audio_samples, include_fourier=self.use_fourier)

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
        if config.dataset == 'wav':
            coord = tf.train.Coordinator()
            reader = self.load_wav(coord)

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
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            reader.start_threads(self.sess)
            num_per_epoch = self.audio_params['num_per_epoch'] 
        #G
        try:
            for epoch in range(config.epoch):
                if config.dataset == 'wav':
                    batch_idxs = min(num_per_epoch, reader.corpus_size) // config.batch_size

                for idx in range(0, batch_idxs):
                    #G
                    if config.dataset == 'wav':
                        pass

                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
                    #G
                    if config.dataset == 'wav':
                        audio_batch = reader.dequeue(self.batch_size) 
                        audio_batch = audio_batch.eval()
                        # Update D network
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                            feed_dict={ self.audio_samples: audio_batch, self.z: batch_z })
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
                        errD_real = self.d_loss_real.eval({self.audio_samples: audio_batch})
                        errG = self.g_loss.eval({self.z: batch_z})
                        #G average over batch
                        D_real = self.D.eval({self.audio_samples: audio_batch}).mean()
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
                                feed_dict={self.z: batch_z, self.audio_samples: audio_batch}
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
                coord.request_stop()
                coord.join(threads)

    def discriminator(self, audio_sample, y=None, reuse=False, include_fourier=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
#        h = conv_bn_lrelu_layer(audio_sample, self.df_dim, name='d_h0_conv')
        i = 2

        for l in range(0, self.num_d_layers+1):
            var_name = 'd_h' + str(l) + '_conv'
            if l == 0:
                h = conv_bn_lrelu_layer(audio_sample, self.df_dim, name=var_name)
            else:
                h = conv_bn_lrelu_layer(h, self.df_dim*i, self.dbn[l-1], name=var_name)
            i *= 2
#        h0 = lrelu(conv1d(audio_sample, self.df_dim, name='d_h0_conv'))
#        h1 = conv_bn_lrelu_layer(h0, self.df_dim*2, self.dbn[0], name='d_h1_conv')
#        h1 = lrelu(self.dbn[0](conv1d(h0, self.df_dim*2, name='d_h1_conv')))
#        h2 = conv_bn_lrelu_layer(h1, self.df_dim*4, self.dbn[1], name='d_h2_conv')
#        h2 = lrelu(self.dbn[1](conv1d(h1, self.df_dim*4, name='d_h2_conv')))
#        h3 = conv_bn_lrelu_layer(h2, self.df_dim*8, self.dbn[2], name='d_h3_conv')
#        h3 = lrelu(self.dbn[2](conv1d(h2, self.df_dim*8, name='d_h3_conv')))
        if self.use_disc:
            h_disc = mb_disc_layer(tf.reshape(h, [self.batch_size, -1]),name='mb_disc')
            h = linear(h_disc, 1, 'd_h3_lin')
        else:
            h = linear(tf.reshape(h, [self.batch_size, -1]), 1, 'd_h3_lin')

        if include_fourier:
            fourier_sample = get_fourier(audio_sample)
#            h_f = conv_bn_lrelu_layer(fourier_sample, self.df_dim, name='d_h0_f_conv')
#            h0_f = lrelu(conv1d(fourier_sample, self.df_dim, name='d_h0_f_conv'))
            i = 2

            for l in range(0, self.num_d_layers+1):
                var_name = 'd_h' + str(l) + '_f_conv'
                if l == 0:
                    h_f = conv_bn_lrelu_layer(fourier_sample, self.df_dim, name=var_name)
                else:
                    h_f = conv_bn_lrelu_layer(h_f, self.df_dim*i, self.dbn[l-1], name=var_name)
                i *= 2
#            h1_f = lrelu(self.d_bn1f(conv1d(h0_f, self.df_dim*2, name='d_h1_f_conv')))
#            h2_f = lrelu(self.d_bn2f(conv1d(h1_f, self.df_dim*4, name='d_h2_f_conv')))
#            h3_f = lrelu(self.d_bn3f(conv1d(h2_f, self.df_dim*8, name='d_h3_f_conv')))
            #import IPython; IPython.embed()
            if self.use_disc:
                h_f_disc = mb_disc_layer(tf.reshape(h_f, [self.batch_size, -1]),name='f_mb_disc')
                h_f = linear(h_f_disc, 1, 'd_h3_f_lin')
            else:
                h_f = linear(tf.reshape(h_f, [self.batch_size, -1]), 1, 'd_h3_f_lin')
            #h5 = linear(tf.concat(1,[h4,h4_f]),1, 'd_h5')
            h = (h+h_f)/2 #average of fourier discrimination and output discrimination

        return tf.nn.sigmoid(h), h

    def generator(self, z, y=None):

        s = self.output_length
        # have to manually set these dimensions to ensure proper run occurs
        sl = [int(s/2), int(s/4), int(s/8), int(s/16)]
        gf_sh = [8, 4, 2, 1]
        strides = [2, 2, 2, 2]

        assert(len(strides) == self.num_g_layers, "Mismatch between number of strides and number of deconv layers")

        # project `z` and reshape
        self.z_, h0_w, h0_b = linear(z, self.gf_dim*gf_sh[0]*sl[-1], 'g_h0_lin', with_w=True)
        h0 = tf.reshape(self.z_, [-1, sl[-1], self.gf_dim * gf_sh[0]])
        a = tf.nn.relu(self.gbn[0](h0))
        self.H, self.H_W, self.H_B = [h0], [h0_w], [h0_b]

        k = -2
        for l in range(self.num_g_layers):
            var_name = 'g_h' + str(l+1)
            a, h, h_w, h_b = deconv_bn_relu_layer(a, [self.batch_size, int(sl[k]), int(self.gf_dim*gf_sh[l+1])],
                                                    self.gbn[l+1], d_w=strides[l], name=var_name, with_w=True)
            self.H.append(h); self.H_W.append(h_w); self.H_B.append(h_b)
            k -= 1

        var_name = 'g_h'+str(self.num_g_layers+1)
        h, h_w, h_b = deconv1d(a, [self.batch_size, s, self.c_dim], d_w=strides[-1], name=var_name, with_w=True)
        self.H.append(h); self.H_W.append(h_w); self.H_B.append(h_b)

        return tf.nn.tanh(h)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        s = self.output_length

        # have to manually set these dimensions to ensure proper run occurs
        sl = [int(s/2), int(s/4), int(s/8), int(s/16)]
        gf_sh = [8, 4, 2, 1]
        strides = [2, 2, 2, 2]

        h0 = tf.reshape(linear(z, self.gf_dim*gf_sh[0]*sl[-1], 'g_h0_lin'),
                        [-1, sl[-1], self.gf_dim * gf_sh[0]])
        a = tf.nn.relu(self.gbn[0](h0, train=False))

        k = -2
        for l in range(self.num_g_layers):
            layer_name = 'g_h' + str(l+1)
            a, _ = deconv_bn_relu_layer(a, [self.batch_size, int(sl[k]), int(self.gf_dim*gf_sh[l+1])],
                                                    self.gbn[l+1], d_w=strides[l], name=layer_name, is_train=False)
            k -= 1

        layer_name = 'g_h'+str(self.num_g_layers+1)
        h = deconv1d(a, [self.batch_size, s, self.c_dim], d_w=strides[-1], name=layer_name)

        return tf.nn.tanh(h)

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
