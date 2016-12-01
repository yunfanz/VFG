from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import json, sys
from audio_reader import AudioReader
from wavenet import WaveNetModel, optimizer_factory, mu_law_encode, mu_law_decode
from ops import *
from utils import *
from postprocess import *

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=1, sample_length=1024,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64, run_g=2, wavenet_params=None,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='default', data_dir=None,
                 audio_params=None, checkpoint_dir=None, wavemodel='./wavenet.ckpt', out_dir=None, use_disc=False, use_fourier=True, mode='generate'):
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
        self.batch_size = batch_size
        #self.sample_size = sample_size
        #self.output_length = output_length
        self.sample_length = sample_length
        self.output_length = sample_length
        self.x_dim = sample_length
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.run_g = run_g
        self.c_dim = c_dim
        self.out_dir = out_dir

        # batch normalization : deals with poor initialization helps gradient flow
        self.q_bn1 = batch_norm(name='q_bn1')
        self.q_bn2 = batch_norm(name='q_bn2')
        self.q_bn3 = batch_norm(name='q_bn3')
        self.q_bn4 = batch_norm(name='q_bn4')

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
            with open(audio_params, 'r') as f:
                self.audio_params = json.load(f)
            self.gf_dim = self.audio_params['gf_dim']
            self.df_dim = self.audio_params['df_dim']
        else:
            self.gf_dim = gf_dim
            self.df_dim = df_dim
        self.wavenet_params = wavenet_params
        self.q_chans = self.wavenet_params["quantization_channels"]
        self.checkpoint_dir = checkpoint_dir
        self.use_disc = use_disc
        self.use_fourier = use_fourier
        self.wavemodel = wavemodel
        if mode == 'train': self.build_model(self.dataset_name)

    def build_model(self, dataset):
        # if self.y_dim:
        #     self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        #G
        if dataset == 'wav':
            self.coord = tf.train.Coordinator()
            self.reader = self.load_wav(self.coord)
            self.audio_batch = self.reader.dequeue(self.batch_size)
            input_batch = mu_law_encode(self.audio_batch, self.q_chans)
            self.hot_batch = one_hot(input_batch, self.batch_size, self.q_chans)
            self.batch_flatten = tf.reshape(self.hot_batch, [self.batch_size, -1])

        # self.z = tf.placeholder(tf.float32, [None, self.z_dim, 1],
        #                         name='z')
        self.z_mean, self.z_log_sigma_sq = self.encoder()
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z_sum_mean = tf.histogram_summary("z_mean", self.z_mean)
        self.z_sum_sig = tf.histogram_summary("z_sig", self.z_log_sigma_sq)
        self.z_sum = tf.merge_summary([self.z_sum_mean, self.z_sum_sig])

        self.g_net = WaveNetModel(
            batch_size=self.batch_size,
            dilations=self.wavenet_params["dilations"],
            filter_width=self.wavenet_params["filter_width"],
            residual_channels=self.wavenet_params["residual_channels"],
            dilation_channels=self.wavenet_params["dilation_channels"],
            skip_channels=self.wavenet_params["skip_channels"],
            quantization_channels=self.wavenet_params["quantization_channels"],
            use_biases=self.wavenet_params["use_biases"],
            scalar_input=self.wavenet_params["scalar_input"],
            initial_filter_width=self.wavenet_params["initial_filter_width"],
            histograms='False', 
            global_scope='gwave')
        
        self.net_in = self.decoder(self.z)
        self.G = self.generator(self.net_in)
        self.sampler = self.generator(self.decoder(self.z, reuse=True))
        self.batch_vae_flatten = tf.reshape(self.net_in, [self.batch_size, -1])
        self.batch_wave_flatten = tf.reshape(self.G, [self.batch_size, -1])

        # with tf.device('/gpu:0'):
        #     """d_net is pretrained model used for feature extraction"""
        #     self.d_net = WaveNetModel(
        #         batch_size=self.batch_size,
        #         dilations=self.wavenet_params["dilations"],
        #         filter_width=self.wavenet_params["filter_width"],
        #         residual_channels=self.wavenet_params["residual_channels"],
        #         dilation_channels=self.wavenet_params["dilation_channels"],
        #         skip_channels=self.wavenet_params["skip_channels"],
        #         quantization_channels=self.wavenet_params["quantization_channels"],
        #         use_biases=self.wavenet_params["use_biases"],
        #         scalar_input=self.wavenet_params["scalar_input"],
        #         initial_filter_width=self.wavenet_params["initial_filter_width"],
        #         histograms='False', 
        #         global_scope='wavenet')
            # self.g_loss = self.discriminator(self.G)
            # self.t_loss = self.target_gen(with_p=False)
        # self.D, self.D_logits = self.discriminator(audio_batch, include_fourier=self.use_fourier)

        # #self.sampler = tf.stop_gradient(self.sampler(self.z))
        # self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True, include_fourier=self.use_fourier)

        
        # #import IPython; IPython.embed()
        # self.d_sum = tf.histogram_summary("d", self.D)
        # self.d__sum = tf.histogram_summary("d_", self.D_)
        # #G need to check sample rate
        # self.G_sum = tf.audio_summary("G", self.G, sample_rate=self.audio_params['sample_rate'])

        # self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake

        # #G experiments with losses
        # #self.max_like_g_loss = tf.reduce_mean(-tf.exp(self.D_logits_)/2.)
        # #self.g_loss = self.max_like_g_loss
        # #import IPython; IPython.embed()
        # #self.g_loss = self.g_loss - self.d_loss

        # self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        # self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        # self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.create_loss_terms()
        #self.balanced_loss = 1. * self.g_loss + 1.* self.vae_loss # can try to weight these.
        self.v_sum = tf.merge_summary([self.vr_loss_sum, self.l_loss_sum, self.vae_loss_sum, self.z_sum])
        self.w_sum = tf.merge_summary([self.wr_loss_sum, self.g_loss_sum])

        self.t_vars = tf.trainable_variables()

        self.q_vars = [var for var in self.t_vars if 'q_' in var.name]
        self.g_vars = [var for var in self.t_vars if 'g_' in var.name]
        #self.dwave_vars = [var for var in t_vars if 'wavenet' in var.name]
        self.gwave_vars = [var for var in t_vars if 'gwave' in var.name]

        self.vae_vars = self.q_vars+self.g_vars
        self.gen_vars = self.g_vars+self.gwave_vars
        self.v_optim = tf.train.AdamOptimizer(2e-5, beta1=self.audio_params['beta1']) \
                          .minimize(self.vae_loss, var_list=self.vae_vars)
        self.gen_optim = tf.train.AdamOptimizer(2e-5, beta1=self.audio_params['beta1']) \
                          .minimize(self.g_loss, var_list=self.gen_vars)
        # g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
        #                   .minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
    def create_loss_terms(self):

        self.vae_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.net_in, self.hot_batch))
        self.wave_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.G, self.hot_batch))
        self.latent_loss = -0.5 * tf.reduce_mean(1 + self.z_log_sigma_sq - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.vae_loss = tf.reduce_mean(self.vae_r_loss + self.latent_loss) #/ self.x_dim 
        self.vr_loss_sum = tf.scalar_summary("vr_loss", self.vae_r_loss)
        self.wr_loss_sum = tf.scalar_summary("wr_loss", self.wave_r_loss)
        self.l_loss_sum = tf.scalar_summary(["l_loss"], self.latent_loss)
        self.vae_loss_sum = tf.scalar_summary("vae_loss", self.vae_loss)
        self.g_loss = self.wave_r_loss #/ self.x_dim
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)

    # def create_gan_loss_terms(self):
    #     # Define loss function and optimiser
    #     self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,tf.ones_like(self.D)))
    #     self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    #     self.d_loss = 1.0*(self.d_loss_real + self.d_loss_fake)/2.
    #     self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    #     self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
    #     self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
    #     self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
    #     self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)


    def generate(self, config):
        '''generate samples from trained model'''
        self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        for counter in range(config.gen_size):
            # batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
            #                         .astype(np.float32)
            samples = self.sess.run(self.sampler, feed_dict={self.z: self.audio_batch.eval()})
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

    def sample(self, counter):
        '''generate samples from trained model'''
        tf.get_variable_scope().reuse_variables()
        _onehot = self.sess.run(self.sampler)
        samples = np.asarray([np.random.choice(self.q_chans,p=_onehot[0][i]) for i in range(self.sample_length)])
        samples = mu_law_decode(samples, self.q_chans).eval()
        samples = samples.reshape((1,self.sample_length))
        file_str = '{:03d}'.format(counter)

        save_audios(samples[0], self.out_dir+'/samples/'+file_str+'.wav', 
            format='.wav', sample_rate=self.audio_params['sample_rate'])
        save_waveform(samples, self.out_dir+'/samples/'+file_str, title='')

        wav_sum_str = tf.audio_summary('S',samples, sample_rate=self.audio_params['sample_rate'],max_outputs=10).eval()
        self.writer.add_summary(wav_sum_str, counter)
        im_sum = get_im_summary(samples, title=file_str)
        summary_str = self.sess.run(im_sum)
        self.writer.add_summary(summary_str, counter)
        
    def encoder(self):
        h5_dim = self.sample_length*self.df_dim//64
        h0 = lrelu(conv1d(self.hot_batch, self.df_dim, name='q_h0_conv'))
        h1 = lrelu(self.q_bn1(conv1d(h0, self.df_dim*2, name='q_h1_conv')))
        h2 = lrelu(self.q_bn2(conv1d(h1, self.df_dim*4, name='q_h2_conv')))
        h3 = lrelu(self.q_bn3(conv1d(h2, self.df_dim*8, name='q_h3_conv')))
        h4 = lrelu(self.q_bn4(conv1d(h3, self.df_dim*16, name='q_h4_conv')))
        z_mean = linear(tf.reshape(h4, [self.batch_size, -1]), self.z_dim, name='q_mean_lin',missing_dim=h5_dim)
        z_log_sigma_sq = linear(tf.reshape(h4, [self.batch_size, -1]), self.z_dim, name='q_sig_lin',missing_dim=h5_dim)

        return (z_mean, z_log_sigma_sq)

    def decoder(self, x, reuse=False):
        tr=True
        if reuse:
            tr=False
            tf.get_variable_scope().reuse_variables()

        s = self.x_dim
        sh = [s//2, s//4, s//8, s//16, s//32, int(s/32/4**1), int(s/32/4**2), int(s/32/4**3)]
        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(self.z, self.gf_dim*16*sh[-1], 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, sh[-1], self.gf_dim * 16])
        h0 = tf.nn.relu(self.g_bn0(self.h0, train=tr))

        self.h1, self.h1_w, self.h1_b = deconv1d(h0, 
            [self.batch_size, sh[-2], self.gf_dim*8], d_w=4, name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1, train=tr))

        h2, self.h2_w, self.h2_b = deconv1d(h1,
            [self.batch_size, sh[-3], self.gf_dim*4], d_w=4, name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2, train=tr))

        h3, self.h3_w, self.h3_b = deconv1d(h2,
            [self.batch_size, sh[-4], self.gf_dim*2], d_w=4, name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3, train=tr))

        h4, self.h4_w, self.h4_b = deconv1d(h3,
            [self.batch_size, sh[-5], self.gf_dim*1], d_w=2, name='g_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4, train=tr))
        h5, self.h5_w, self.h5_b = deconv1d(h4,
            [self.batch_size, sh[-6], self.gf_dim*1], d_w=2, name='g_h5', with_w=True)
        h5 = tf.nn.relu(self.g_bn5(h5, train=tr))
        h6, self.h6_w, self.h6_b = deconv1d(h5,
            [self.batch_size, sh[-7], self.gf_dim*1], d_w=2, name='g_h6', with_w=True)
        h6 = tf.nn.relu(self.g_bn6(h6, train=tr))
        h7, self.h7_w, self.h7_b = deconv1d(h6,
            [self.batch_size, sh[-8], self.gf_dim*1], d_w=2, name='g_h7', with_w=True)
        h7 = tf.nn.relu(self.g_bn7(h7, train=tr))
        h8, self.h8_w, self.h8_b = deconv1d(h7,
            [self.batch_size, s, self.q_chans], d_w=2, name='g_h8', with_w=True)
        h_norm = tf.reduce_sum(h8, 2, keep_dims=True)
        h_prob = h8/h_norm 
        return h_prob

    def generator(self, x, y=None):

        h_prob = tf.exp(self.g_net.run(x, encoded=True))
        h_norm = tf.reduce_sum(h_prob, 2, keep_dims=True)
        h_prob = h_prob/h_norm 
        return h_prob





    def train(self, config):
        """Train DCGAN"""
        #G
        # if config.dataset == 'wav':
        #     coord = tf.train.Coordinator()
        #     reader = self.load_wav(coord)
        #import IPython; IPython.embed()

        #import IPython; IPython.embed()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        # self.sess.run(self.d_net.init_ops)

        # variables_to_restore = {
        #     var.name[:-2]: var for var in tf.all_variables()
        #     if 'wavenet' in var.name and not ('state_buffer' in var.name or 'pointer' in var.name)}
        # saver = tf.train.Saver(variables_to_restore)

        # print('Loading discriminator wavenet from {}'.format(self.wavemodel))
        # saver.restore(self.sess, self.wavemodel)


        # self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
        #     self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        
        self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        # sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

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

                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim, 1]) \
                                .astype(np.float32)

                    #G
                    if config.dataset == 'wav':
                        # audio_batch = reader.dequeue(self.batch_size) 
                        # audio_batch = audio_batch.eval()
                        # Update D network
                        # _, summary_str = self.sess.run([d_optim, self.d_sum],
                        #     feed_dict={ self.z: batch_z })
                        # self.writer.add_summary(summary_str, counter)

                        _, summary_str = self.sess.run([self.v_optim, self.v_sum])
                        self.writer.add_summary(summary_str, counter)
                        _, summary_str = self.sess.run([self.gen_optim, self.w_sum])
                        self.writer.add_summary(summary_str, counter)
                        errV = self.vae_loss.eval()
                        errG = self.g_loss.eval()
                        #import IPython; IPython.embed()


                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f vae_loss: %.6f" \
                        % (epoch+1, idx+1, batch_idxs,
                            time.time() - start_time, errG, errV))
                    if np.mod(counter, config.print_every) == 1:
                        sys.stdout.flush()

                    if np.mod(counter, config.save_every) == 1:
                        print('saving sample')
                        self.sample(counter)
                    #     #G
                    #     if config.dataset == 'wav':
                    #         # samples, d_loss, g_loss = self.sess.run(
                    #         #     [self.sampler, self.d_loss, self.g_loss],
                    #         #     feed_dict={self.z: sample_z, self.images: sample_images.eval()}
                    #         # )
                    #         samples, d_loss, g_loss = self.sess.run(
                    #             [self.sampler, self.d_loss, self.g_loss],
                    #             feed_dict={self.z: batch_z}
                    #         )
                    #         #import IPython; IPython.embed()
                    #     # Saving samples
                    #     if config.dataset == 'wav':
                    #         im_title = "d_loss: %.5f, g_loss: %.5f" % (d_loss, g_loss)
                    #         file_str = '{:02d}_{:04d}'.format(epoch, idx)
                    #         save_waveform(samples,config.out_dir+'/samples/train_'+file_str, title=im_title)
                    #         im_sum = get_im_summary(samples, title=file_str+im_title)
                    #         summary_str = self.sess.run(im_sum)
                    #         self.writer.add_summary(summary_str, counter)
                            
                    #         save_audios(samples[0], config.out_dir+'/samples/train_'+file_str+'.wav', 
                    #             format='.wav', sample_rate=self.audio_params['sample_rate'])
                    #     print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

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

    # def discriminator(self, encoded_sample):
    #     # if reuse:
    #     #     tf.get_variable_scope().reuse_variables()
    #     dwave_loss = self.d_net.loss(encoded_sample, encoded=True)
    #     return dwave_loss
    # def target_gen(self, with_p=False):

    #     s = self.audio_batch
    #     t_prob = tf.exp(self.d_net.run(s))
    #     t_norm = tf.reduce_sum(t_prob, 2, keep_dims=True)
    #     t_prob = t_prob/t_norm 
    #     t_loss = self.d_net.loss(t_prob, encoded=True)
    #     if with_p:
    #         return (t_prob, t_loss)
    #     else: 
    #         return t_loss


    # def sampler(self, z, y=None):
    #     tf.get_variable_scope().reuse_variables()
    #     h_wave = self.g_net.run(z)
    #     prediction = [np.random.choice(
    #         np.arange(self.wavenet_params["quantization_channels"]), \
    #         p=h_wave[0][i]) for i in range(self.sample_length)]
    #     prediction = tf.argmax(h_wave, 2)

    #     waveform = mu_law_decode(prediction, self.g_net.quantization_channels)
    #     return waveform

    

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
