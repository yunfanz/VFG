from toolz import compose
from functools import partial 
import numpy as np
import librosa
#G written with the goal to minimize memory usage

def _complexify(real, imag):
	return real+1j*imag
def _get_shifted(spectrogram):
	return np.fft.ifftshift(np.fft.ifft(spectrogram), axes=1).real.astype(np.float16)
def _realign(shifted, stride=1):
	leng = shifted.shape[1]
	#return np.array([shifted[i][leng//2-i*stride:leng-i*stride] for i in range(leng//2//stride)])
	return np.array([shifted[i][leng//2-i*stride:leng-i*stride] for i in range(leng//2//stride)])

def despectrofy(real, imag, stride=1):
	return compose(partial(_realign, stride=stride), _get_shifted, _complexify)(real, imag)

def save_wav(waveform, filename, sample_rate=8000):
    '''Save the audio into wav file (adapted from tensorflow-wavenet)
    '''
    #y = np.array(waveform)
    librosa.output.write_wav(filename, waveform, sample_rate)



