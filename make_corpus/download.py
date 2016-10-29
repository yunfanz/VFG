#modified from http://willdrevo.com/downloading-youtube-and-soundcloud-audio-with-python-and-pandas/
import youtube_dl
import pandas as pd
import os
import traceback
import argparse
CSV = "videos.csv"
parser = argparse.ArgumentParser(description='Download dataset from youtube.')
parser.add_argument('--csv', type=str, dest=CSV, default="video.csv",
                        help='The csv file containing links to download.')

# create directory
savedir = "raw"
if not os.path.exists(savedir):
    os.makedirs(savedir)

def make_savepath(description, savedir=savedir):
    return os.path.join(savedir, "%s.wav" % (description))

# create YouTube downloader
options = {
    'format': 'bestaudio/best', # choice of quality
    'extractaudio' : True,      # only keep the audio
    'audioformat' : "wav",      # convert to wav 
    'outtmpl': '%(id)s',        # name the file the ID of the video
    'noplaylist' : True,}       # whether only download single song, or playlist
ydl = youtube_dl.YoutubeDL(options)

with ydl:
    # read in videos CSV with pandas
    df = pd.read_csv(CSV, sep=";", skipinitialspace=True)
    df.Link = df.Link.map(str.strip)  # strip space from URLs

    # for each row, download
    for _, row in df.iterrows():
        print("Downloading: %s" % (row.DESCR))

        # download location, check for progress
        savepath = make_savepath(row.DESCR)
        try:
            os.stat(savepath)
            print("%s already downloaded, continuing..." % savepath)
            continue

        except OSError:
            # download video
            try:
                result = ydl.extract_info(row.Link, download=True)
                os.rename(result['id'], savepath)
                print("Downloaded and converted %s successfully!" % savepath)

            except Exception as e:
                print("Can't download audio! %s\n" % traceback.format_exc())