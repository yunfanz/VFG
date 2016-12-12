import numpy as np
import pandas as pd
import json
import os
import math
import fnmatch
from os.path import splitext # split the extension of a file
import argparse
parser = argparse.ArgumentParser(description='Cut the audio files from the corpus into samples')
parser.add_argument('--out', metavar='N', type=str, nargs='+', help='output directory.')
parser.add_argument('--format', metavar='N', type=str, nargs='+', choices=['wav', 'ogg', 'mp3', 'raw'],
                   help='format of the files to be preprocessed. Formats other than \'wav\' '
                   + '\'raw\' need ffmpeg library.')
parser.add_argument('--sample_len', metavar='N', type=int, nargs='+',
                   help='length of the sample to be cut. DEFAULT: 1<sec>')
parser.add_argument('--sample_rate', metavar='N', type=int, nargs='+',
                   help='length of the sample to be cut. DEFAULT: 1<sec>')

def make_wav(waveform, filename, sample_rate):
    '''Save the audio into wav file (adapted from tensorflow-wavenet)
    '''
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)

def make_singlemode_corpus(nfiles, outdir, sample_rate, sample_length):
	om = np.random.rand(nfiles)*2*np.pi*20
	phi = np.random.rand(nfiles)*2*np.pi
	t = np.linspace(0, sample_length/sample_rate, sample_length).reshape((1,-1))
	Y = np.sin(om*t+phi)
	for n,y in enumerate(Y):
		make_wav(y, outdir+'/'+str(n), sample_rate)


if __name__ == '__main__':
    args = parser.parse_args()
    # do not support multi directories for now.
    out_dir = args.out[0]
    sample_len = args.sample_len[0]
    sample_rate = args.sample_rate[0]
    make_singlemode_corpus()