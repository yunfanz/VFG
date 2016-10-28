#####################################################################
# PREPROCESS.PY
# Given a folder of wav files, generate smaller duration wav samples
# by slicing the original
#####################################################################

# IMPORTS
import numpy as np
from pydub import AudioSegment
import os
import math
import fnmatch
from os.path import splitext # split the extension of a file
import argparse

# PARSER
parser = argparse.ArgumentParser(description='Cut the audio files from the corpus into samples')
parser.add_argument('corpus', metavar='N', type=str, nargs='+', 
					help='directory of the corpus.')
parser.add_argument('--out', metavar='N', type=str, nargs='+', help='output directory. DEFAULT: ./preprocessed')
parser.add_argument('--format', metavar='N', type=str, nargs='+', choices=['wav', 'ogg', 'mp3', 'raw'],
                   help='format of the files to be preprocessed. Formats other than \'wav\' '
                   + '\'raw\' need ffmpeg library.')
parser.add_argument('--sample_len', metavar='N', type=int, nargs='+',
                   help='length of the sample to be cut')

SEC_TO_MILLISEC = 1000

def process_audio(in_path, ext, out_path='./preprocessed', sample_len=10):
	'''
	This function loads audio files from \'path\'.
	@param in_path: the directory of the corpus dir
	@param ext: the format of the files
	@param out_path: the directory of the output dir. DEFAULT: ./preprocessed
	@param sample_len: length of the sample that to be cut from the original audio files \
						in seconds
	'''
	if not os.path.exists(in_path):
		print('The corpus folder does not exist. Ending...')
		return
	
	if out_path is None or len(out_path) == 0:
		out_path = './preprocessed'
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	
	pattern = '*.' + ext
	for root, dirnames, filenames in os.walk(in_path):
	    for fn in fnmatch.filter(filenames, pattern):
	    	filename, _ = os.path.splitext(fn)
	        sound = AudioSegment.from_file(os.path.join(root, filename) + '.' + ext, format=ext)
	        
	        len_in_sec = sound.duration_seconds
	        
	        if len_in_sec == 0:
	        	print('zero-length file encountered: ' + str(filename) + '\n Passed.')
	        	continue
	        
	        num_samples = int(math.ceil(len_in_sec/sample_len))
	        for i in range(num_samples-1):
	        	begin = i*sample_len*SEC_TO_MILLISEC
	        	end = (i+1)*sample_len*SEC_TO_MILLISEC
	        	sample = sound[begin:end]
	        	out_name = filename + '_sample_' + str(i) + '.' + ext
	        	out = os.path.join(out_path, out_name)
	        	sample.export(out, format=ext)

	        out_name = filename + '_sample_' + str(num_samples-1) + '.' + ext
	        out = os.path.join(out_path, out_name)
	        sample = sound[(num_samples-1)*sample_len*SEC_TO_MILLISEC:] # last sample
	        sample.export(out, format=ext)


if __name__ == '__main__':
    args = parser.parse_args()
    # do not support multi directories for now.
    in_dir = args.corpus[0]
    out_dir = args.out[0]
    ext = args.format[0]
    sample_len = args.sample_len[0]
    process_audio(in_path=in_dir, ext=ext, out_path=out_dir, sample_len=sample_len)