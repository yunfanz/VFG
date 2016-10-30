#####################################################################
# PREPROCESS.PY
# Given a folder of wav files, generate smaller duration wav samples
# by slicing the original. Also supports downloading directly from
# youtube.com
# usage:
# python preprocess.py --youtube youtube.json --out ./preprocessed --format mp3 --sample_len 1
# if sample_len == 0, then no slicing
#####################################################################

# IMPORTS
import numpy as np
from pydub import AudioSegment
import youtube_dl
import pandas as pd
import traceback
import json
import os
import math
import fnmatch
from os.path import splitext # split the extension of a file
import argparse

# PARSER
parser = argparse.ArgumentParser(description='Cut the audio files from the corpus into samples')
parser.add_argument('--youtube', metavar='N', type=str, nargs='+',
                    help='json file containing params for a youtube download. DEFAULT: ./youtube.json')
parser.add_argument('--corpus', metavar='N', type=str, nargs='+', 
                    help='directory of the corpus. If youtube argument is y, then a web address should be passed in.')
parser.add_argument('--out', metavar='N', type=str, nargs='+', help='output directory.')
parser.add_argument('--format', metavar='N', type=str, nargs='+', choices=['wav', 'ogg', 'mp3', 'raw'],
                   help='format of the files to be preprocessed. Formats other than \'wav\' '
                   + '\'raw\' need ffmpeg library.')
parser.add_argument('--sample_len', metavar='N', type=int, nargs='+',
                   help='length of the sample to be cut. DEFAULT: 1<sec>')

SEC_TO_MILLISEC = 1000

def make_savepath(title, artist, savedir='./data/raw', format='wav'):
    return os.path.join(savedir, "%s--%s.%s" % (title, artist, format))


def download(config='./youtube.json'):
    '''
    This function downloads audio files from the specified youtube address
    @param config: json file for the youtube download
    @param out_path: temporary storage location for the downloaded file
    '''
    #modified from http://willdrevo.com/downloading-youtube-and-soundcloud-audio-with-python-and-pandas/
    with open(config, 'r') as a:
        dl_config = json.load(a)
    CSV = dl_config['source'] # a csv file
    savedir = dl_config['savedir']
    music_config = dl_config['music_config']

    if not os.path.exists(savedir):
        os.makedirs(savedir)    
    with open(music_config, 'r') as mc:
        ydl_opts = json.load(mc)

    ext = ydl_opts['audio-format']
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        df = pd.read_csv(CSV, sep=";", skipinitialspace=True)
        df.Link = df.Link.map(str.strip)  # strip space from URLs
        print(df['Link'])

        # for each row, download
        for _, row in df.iterrows():
            print("Downloading: %s from %s..." % (row.Title, row.Link))
            # download location, check for progress
            savepath = make_savepath(row.Title, row.Artist, savedir=savedir, format=ext)
            try:
                os.stat(savepath)
                print("%s already downloaded, continuing..." % savepath)
                continue
            except OSError:
                # download video
                try:
                    #ydl.download([row.Link])
                    result = ydl.extract_info(row.Link, download=True)
                    os.rename(result['id'] + '.' + ext, savepath)
                    print("Downloaded and converted %s successfully!" % savepath)
                except Exception as e:
                    print("Can't download audio! %s\n" % traceback.format_exc())
    return savedir, ext


def process_audio(in_path, ext, out_path, sample_len=10):
    '''
    This function loads audio files from \'path\'.
    @param in_path: the directory of the corpus dir
    @param ext: the format of the files
    @param out_path: the directory of the output dir. DEFAULT: ./preprocessed
    @param sample_len: length of the sample that to be cut from the original audio files \
                        in seconds
    '''
    # no preprocessing done if sample_len is 0
    if sample_len == 0:
        return

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
    out_dir = args.out[0]
    sample_len = args.sample_len[0]
    if args.youtube is not None:
        in_dir, ext = download(args.youtube[0])
    else:
        in_dir = args.corpus[0]
        ext = args.ext[0]
    process_audio(in_path=in_dir, ext=ext, out_path=out_dir, sample_len=sample_len)
