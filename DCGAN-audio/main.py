import os
import scipy.misc
import numpy as np
import audio_reader
from model import DCGAN
from utils import pp, visualize, to_json
import tensorflow as tf
import json
import warnings
from datetime import datetime
flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("save_every", 100, "save sample outputs every [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The number of train waveforms [np.inf]")
flags.DEFINE_integer("sample_size", 1, "The size of sample batch waveforms [1]")
flags.DEFINE_integer("batch_size", 1, "The size of input batch waveforms [1]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_length", 4096, "The length of the output waveform to produce ")
flags.DEFINE_integer("sample_length", 4096, "The length of the input waveforms; has to match output_length if is_train")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("z_dim", 100, "length of latent code. [100]")
flags.DEFINE_string("dataset", "wav", "The name of dataset ['wav']")
flags.DEFINE_string("data_dir", None, "Optional path to data directory")
flags.DEFINE_string("out_dir", None, "Directory name to save the output samples, checkpoint, log")
flags.DEFINE_string("checkpoint_dir", None, "Directory name to LOAD checkpoint, new checkpoints will be saved to out_dir/checkpoint")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
#flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("use_disc", False, "Whether to use mini-batch discrimination on the discriminator")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("audio_params", './audio_params.json', 'JSON file with tune-specific parameters.')
FLAGS = flags.FLAGS

def main(_):
    STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    pp.pprint(flags.FLAGS.__flags)
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/train_'+STARTED_DATESTRING
        print('Using default out_dir {0}'.format(FLAGS.out_dir))
    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = FLAGS.out_dir+'/checkpoint'
    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
        os.makedirs(FLAGS.out_dir+'/samples')
        os.makedirs(FLAGS.out_dir+'/checkpoint')
        os.makedirs(FLAGS.out_dir+'/logs')
    #if not os.path.exists(FLAGS.sample_dir):
    #    os.makedirs(FLAGS.sample_dir)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        #G
        if FLAGS.dataset == 'wav':
            with open('audio_params.json', 'r') as f:
                audio_params = json.load(f)
            FLAGS.epoch = audio_params['epoch']
            FLAGS.learning_rate = audio_params['learning_rate']
            FLAGS.beta1 = audio_params['beta1']
            FLAGS.z_dim = audio_params['z_dim']
            if FLAGS.is_train:
                if FLAGS.sample_length != FLAGS.output_length:
                    #G: may not need sample_length at all
                    warnings.warn('Training sample_length must be equal to output_length')
                    print('Setting sample_length to output_length', FLAGS.output_length)
                    FLAGS.sample_length = FLAGS.output_length
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, sample_size=FLAGS.sample_size, batch_size=FLAGS.batch_size, z_dim=FLAGS.z_dim, 
                    output_length=FLAGS.output_length, sample_length=FLAGS.sample_length, c_dim=1,
                    dataset_name=FLAGS.dataset, audio_params=FLAGS.audio_params, data_dir=FLAGS.data_dir, use_disc=FLAGS.use_disc, checkpoint_dir=FLAGS.checkpoint_dir, out_dir=FLAGS.out_dir)
        else:
            raise Exception('dataset not understood')

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        if FLAGS.visualize:
            to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                                          [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                                          [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                                          [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                                          [dcgan.h4_w, dcgan.h4_b, None])

            # Below is codes for visualization
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
    tf.app.run()
