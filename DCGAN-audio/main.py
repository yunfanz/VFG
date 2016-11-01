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
from shutil import copyfile
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_integer("save_every", 100, "save sample outputs every [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 1, "The size of input batch waveforms [1]")
flags.DEFINE_integer("gen_size", 1, "How many batches to generate (generate mode only)")
flags.DEFINE_integer("sample_length", 4096, "The length of the waveforms, both input and output")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("run_g", 2, "Number of times to update the generator per discriminator update [2]")
#flags.DEFINE_integer("z_dim", 100, "length of latent code. [100]")
flags.DEFINE_string("dataset", "wav", "The name of dataset ['wav']")
flags.DEFINE_string("data_dir", None, "Optional path to data directory")
flags.DEFINE_string("out_dir", None, "Directory name to save the output samples, checkpoint, log")
flags.DEFINE_string("checkpoint_dir", None, "Directory name to LOAD checkpoint, new checkpoints will be saved to out_dir/checkpoint")
flags.DEFINE_string("mode", 'generate', "running mode, has to be either train or generate")
flags.DEFINE_boolean("use_fourier", True, "Whether the discriminator will use Fourier information")
flags.DEFINE_boolean("use_disc", False, "Whether to use mini-batch discrimination on the discriminator")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("audio_params", None, 'JSON file with tune-specific parameters.')
FLAGS = flags.FLAGS

def main(_):
    STARTED_DATESTRING = "{0:%Y-%m-%dT%H:%M:%S}".format(datetime.now()).replace(":", "-")
    pp.pprint(flags.FLAGS.__flags)

    assert FLAGS.mode.lower() in ('train','generate'), "mode must be 'train' or 'generate'!"
    FLAGS.mode = FLAGS.mode.lower()
    if FLAGS.mode == 'train': 
        if FLAGS.out_dir is None:
            FLAGS.out_dir = 'out/train_'+STARTED_DATESTRING
            print('Using default out_dir {0}'.format(FLAGS.out_dir))
        else:
            if FLAGS.out_dir.endswith('/'): FLAGS.out_dir = FLAGS.out_dir[:-1]
        if FLAGS.checkpoint_dir is None:
            FLAGS.checkpoint_dir = FLAGS.out_dir+'/checkpoint'
    else: 
        if FLAGS.checkpoint_dir is None:
            raise Exception('Cannot generate: checkpoint {0} does not exist!'.format(FLAGS.checkpoint_dir))
        else:
            if FLAGS.checkpoint_dir.endswith('/'): FLAGS.checkpoint_dir=FLAGS.checkpoint_dir[:-1]
        if FLAGS.out_dir is None:
            FLAGS.out_dir = 'out/gene_'+STARTED_DATESTRING

    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
        #import IPython; IPython.embed()
        if FLAGS.mode == 'train':
            os.makedirs(FLAGS.out_dir+'/samples')
            os.makedirs(FLAGS.out_dir+'/checkpoint')
            os.makedirs(FLAGS.out_dir+'/logs')

    if FLAGS.audio_params is None:
        if FLAGS.mode == 'train':
            FLAGS.audio_params='./audio_params.json'
            copyfile(FLAGS.audio_params,FLAGS.checkpoint_dir+'/audio_params.json')
        else:
            print('Using json file from {0}'.format(FLAGS.checkpoint_dir))
            FLAGS.audio_params=FLAGS.checkpoint_dir+'/audio_params.json'


    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        #G
        if FLAGS.dataset == 'wav':
            with open('audio_params.json', 'r') as f:
                audio_params = json.load(f)
            FLAGS.epoch = audio_params['epoch']
            FLAGS.learning_rate = audio_params['learning_rate']
            FLAGS.beta1 = audio_params['beta1']
            FLAGS.sample_length = audio_params['sample_length']
            dcgan = DCGAN(sess, batch_size=FLAGS.batch_size, z_dim=audio_params['z_dim'], 
                    sample_length=FLAGS.sample_length, c_dim=1, dataset_name=FLAGS.dataset, audio_params=FLAGS.audio_params, 
                    data_dir=FLAGS.data_dir, use_disc=FLAGS.use_disc, use_fourier=FLAGS.use_fourier,
                    run_g=FLAGS.run_g, checkpoint_dir=FLAGS.checkpoint_dir, out_dir=FLAGS.out_dir, mode=FLAGS.mode)
        else:
            raise Exception('dataset not understood')

        if FLAGS.mode == 'train':
            dcgan.train(FLAGS)
        else:
            print('Generating {0} batches of size {1} from checkpoint {2}'.format(FLAGS.gen_size, FLAGS.batch_size, FLAGS.checkpoint_dir))
            dcgan.load(FLAGS.checkpoint_dir)
            dcgan.generate(FLAGS)

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
