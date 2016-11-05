'''
 Adapted from https://github.com/carpedm20/DCGAN-tensorflow
'''
import fnmatch
import os
import re
import threading
import numpy as np
import tensorflow as tf

#G
def get_corpus_size(directory, pattern='*.npy'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return len(files)

def find_files(directory, pattern='*.npy'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_spectrograms(directory):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        spectro = np.load(filename)
        yield spectro, filename

class NumpyReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 queue_size=64):
        self.audio_dir = audio_dir
        self.coord = coord
        self.corpus_size = get_corpus_size(self.audio_dir)
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_spectrograms(self.audio_dir)
            for spectro, filename in iterator:
                #print(filename)
                if self.coord.should_stop():
                    stop = True
                    break
                spectro = tf.cast(spectro,tf.float32)
                sess.run(self.enqueue, feed_dict={self.sample_placeholder: spectro})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
