import numpy as np
import pandas as pd
import json
import os
import math
import fnmatch
from os.path import splitext # split the extension of a file
import argparse
import librosa
parser = argparse.ArgumentParser(description='Cut the audio files from the corpus into samples')
parser.add_argument('--out', default='./fourier_corpus', help='output directory.')
parser.add_argument('--format', metavar='N', type=str, nargs='+', choices=['wav', 'ogg', 'mp3', 'raw'],
                   help='format of the files to be preprocessed. Formats other than \'wav\' '
                   + '\'raw\' need ffmpeg library.')
parser.add_argument('--sample_len', default=32000,
                   help='length of the sample to be cut. DEFAULT: 1<sec>')
parser.add_argument('--sample_rate', default=16000,
                   help='length of the sample to be cut. DEFAULT: 1<sec>')
parser.add_argument('--fmin', default=80,
                   help='length of the sample to be cut. DEFAULT: 1<sec>')
parser.add_argument('--fmax', default=15000,
                   help='length of the sample to be cut. DEFAULT: 1<sec>')
parser.add_argument('--nfiles', default=100)

def make_wav(waveform, filename, sample_rate):
    '''Save the audio into wav file (adapted from tensorflow-wavenet)
    '''
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)

def make_singlemode_corpus(nfiles, outdir, sample_rate, sample_length, fmin, fmax):
	om = np.random.rand(nfiles).reshape((-1,1))*2*np.pi*(fmax-fmin)+fmin
	phi = np.random.rand(nfiles).reshape((-1,1))*2*np.pi
	t = np.linspace(0, sample_length/sample_rate, sample_length)
	Y = np.sin(om*t+phi).astype(np.float32)
	#import IPython; IPython.embed()
	for n,y in enumerate(Y):
		make_wav(y, outdir+'/'+str(n)+'.wav', sample_rate)


if __name__ == '__main__':
    args = parser.parse_args()
    # do not support multi directories for now.
    
    #import IPython; IPython.embed()
    out_dir = args.out
    sample_len = int(args.sample_len)
    sample_rate = int(args.sample_rate)


    make_singlemode_corpus(int(args.nfiles), out_dir, sample_rate, sample_len, args.fmin, args.fmax)