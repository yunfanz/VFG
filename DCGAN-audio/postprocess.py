import numpy as np
from toolz import compose

def pc_chop(samples, n_mode=200):
	'''leaving only the top n_mode Fourier modes'''
	ifft = compose(np.fft.ifftshift, np.fft.ifft)
	samples = samples[...,0]  #rid of c_dim
	s_ = np.fft.fft(samples) #default along last axis
	order = np.argsort(np.abs(s_),axis=-1)
	for i in range(order.shape[0]):
		s_[i, order[i,:-n_mode]] = 0
	s__ = ifft(s_)
	processed = s__[...,np.newaxis].real.astype('float32')
	return processed
