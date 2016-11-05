import os
import time
from glob import glob
import numpy as np
import fnmatch
from audio_reader import AudioReader
from ops import *
from utils import *
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

SAMPLE_RATE = 8000
SAMPLE_LENGTH = 8192
silence_threshold = 0.3
IN_DIR = '../../VCTK-Corpus/wav48/p225/'
OUT_DIR = './Spectro/'

def basename(path):
	return os.path.splitext(os.path.basename(path))[0]

def get_corpus_size(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return len(files)

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_generic_audio(filename, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.reshape(-1, 1)
    #import IPython; IPython.embed()
    return audio

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def spectrofy(filename, sample_rate, if_return=False):
	print('Processing: ', filename)
	audio = load_generic_audio(filename, sample_rate)
	audio = trim_silence(audio, silence_threshold)
	if audio.size < SAMPLE_LENGTH*2:
		print('Audio silent or too short, skipping')
		return  
	T = np.arange(SAMPLE_LENGTH)
	shifted = np.array([audio[t:t+SAMPLE_LENGTH] for t in T])  #could be faster
	real = np.fft.fft(shifted).real.astype(np.float16)
	imag = np.fft.fft(shifted).imag.astype(np.float16)

	spectrogram = np.stack([real,imag],axis=2)
	np.save(OUT_DIR+basename(filename), spectrogram)
	#print(spectrogram.shape)
	if if_return:
		return spectrogram
	else:
		return 


def make_corpus_cpu(directory=IN_DIR, sample_rate=SAMPLE_RATE):
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)
	files = find_files(directory)
	collect = Parallel(n_jobs=num_cores)(delayed(spectrofy)(fn, sample_rate) for fn in files)


if __name__ == '__main__':
	make_corpus_cpu()


