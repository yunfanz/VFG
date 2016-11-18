#modified from http://willdrevo.com/downloading-youtube-and-soundcloud-audio-with-python-and-pandas/
import youtube_dl
import pandas as pd
import os
import traceback
import argparse
import json
import librosa, fnmatch
CSV = "youtube_source.csv"
sample_rate = 32000
savedir='./raw'
parser = argparse.ArgumentParser(description='Download dataset from youtube.')
parser.add_argument('--csv', type=str, dest=CSV, default="video.csv",
                        help='The csv file containing links to download.')
parser.add_argument('--savedir', type=str, dest=savedir,
                        help='directory to download to')
parser.add_argument('--sample_rate', type=int, dest=sample_rate,
                        help='sample rate to convert to')


def make_savepath(description, savedir=savedir):
    return os.path.join(savedir, "%s.wav" % (description))

def download(CSV=CSV, savedir=savedir, music_config="./music_config.json"):
    '''
    This function downloads audio files from the specified youtube address
    @param config: json file for the youtube download
    @param out_path: temporary storage location for the downloaded file
    '''
    #modified from http://willdrevo.com/downloading-youtube-and-soundcloud-audio-with-python-and-pandas/

    if not os.path.exists(savedir):
        os.makedirs(savedir)    
    with open(music_config, 'r') as mc:
        ydl_opts = json.load(mc)

    ext = ydl_opts['audioformat']
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        df = pd.read_csv(CSV, sep=";", skipinitialspace=True)
        df.Link = df.Link.map(str.strip)  # strip space from URLs

        # for each row, download
        for _, row in df.iterrows():
            # download location, check for progress
            savepath = make_savepath(row.DESC, savedir=savedir)
            try:
                os.stat(savepath)
                print("%s already downloaded, continuing..." % savepath)
                continue
            except OSError:
                # download video
                try:
                    #ydl.download([row.Link])
                    result = ydl.extract_info(row.Link, download=True)
                    os.rename(result['id'], savepath)
                    print("Downloaded: %s of size %d, original format %s " % (result['title'], result['filesize'], result['ext']))
                    #print("Saving %s successfully!" % savepath)
                except Exception as e:
                    print("Can't download audio! %s\n" % traceback.format_exc())
    #return savedir, ext
def to_signed(directory=savedir, pattern='*.wav', sample_rate=sample_rate):
    '''Recursively finds all files matching the pattern.'''
    files = []
    print('Converting files to signed float32, sample_rate', sample_rate)
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            file = os.path.join(root, filename)
            audio, _ = librosa.load(file, sr=sample_rate, mono=True)
            librosa.output.write_wav(file, audio, sample_rate)



if __name__=='__main__':
    download()
    to_signed()