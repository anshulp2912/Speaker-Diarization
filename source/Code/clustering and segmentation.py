# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:10:33 2019

@author: Anshul & Anuj
"""
import os
import aifc
import numpy 
from pydub import AudioSegment
import numpy as np
def read_aif(path):
    """
    Read audio file with .aif extension
    """
    sampling_rate = -1
    signal = np.array([])
    try:
        with aifc.open(path, 'r') as s:
            nframes = s.getnframes()
            strsig = s.readframes(nframes)
            signal = numpy.fromstring(strsig, numpy.short).byteswap()
            sampling_rate = s.getframerate()
    except:
        print("Error: read aif file. (DECODING FAILED)")
    return sampling_rate, signal

def read_audio_generic(path):
    """
    Function to read audio files with the following extensions
    [".mp3", ".wav", ".au", ".ogg"]
    """
    sampling_rate = -1
    signal = np.array([])
    try:
        audiofile = AudioSegment.from_file(path)
        data = np.array([])
        if audiofile.sample_width == 2:
            data = numpy.fromstring(audiofile._data, numpy.int16)
        elif audiofile.sample_width == 4:
            data = numpy.fromstring(audiofile._data, numpy.int32)

        if data.size > 0:
            sampling_rate = audiofile.frame_rate
            temp_signal = []
            for chn in list(range(audiofile.channels)):
                temp_signal.append(data[chn::audiofile.channels])
            signal = numpy.array(temp_signal).T
    except:
        print("Error: file not found or other I/O error. (DECODING FAILED)")
    return sampling_rate, signal

def read_audio_file(path):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = -1
    signal = np.array([])
    extension = os.path.splitext(path)[1].lower()
    if extension in ['.aif', '.aiff']:
        sampling_rate, signal = read_aif(path)
    elif extension in [".mp3", ".wav", ".au", ".ogg"]:
        sampling_rate, signal = read_audio_generic(path)
    else:
        print(f"Error: unknown file type {extension}")

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal



from sklearn.cluster import KMeans
import pandas as pd


[fs, x] = read_audio_file('try.wav')
x1=x
time = np.arange(0, float(x.shape[0]), 1) / fs
song=AudioSegment.from_file('try.wav')
samples=song.get_array_of_samples()
samples=np.array(samples)

x = stereo_to_mono(x)
duration = len(x) / fs
merge_norm_feats=pd.read_csv('mt_feats_norms.csv',header=None)
merge_norm_feats=merge_norm_feats.T

dic=[]
x=[]
for i in range(1,10):
    x.append(i)
    km=KMeans(n_clusters=i)
    km.fit(merge_norm_feats)
    dic.append(km.inertia_)

import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt

plt.figure(figsize=(15,14))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(merge_norm_feats, method='ward'))
plt.savefig('dendograms')

labelList = range(1, 10)


dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
#executing this will give optimal 2 clusters    
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(14,14))
ax=fig.add_subplot(311)
ax.set_ylabel('Inertia')
ax.set_xlabel('no_of_clusters')
ax.plot(x,dic)

optimal_clusters=2

kmean=KMeans(n_clusters=2)
tryi=kmean.fit_predict(merge_norm_feats)
mt_step=0.2

from sklearn.cluster import AgglomerativeClustering 
tryin = AgglomerativeClustering(n_clusters = 2).fit_predict(merge_norm_feats) 
cls=tryin

class_names=[]
for i in range(0,optimal_clusters):
    class_names.append(i)

import matplotlib.pyplot as plt    
fig=plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.set_yticks(np.array(range(len(class_names))))
ax1.axis((0, duration, -1, len(class_names)))
ax1.set_yticklabels(class_names)
ax1.set_ylabel('Speakers')
ax1.set_xlabel('Time')
ax1.plot(np.array(range(len(cls)))*mt_step+mt_step/2.0, cls,color='gray')
fi=np.array(range(len(cls)))*mt_step+mt_step/2.0


ax2=fig.add_subplot(212)
ax2.set_xlabel('Time')
ax2.plot(time,x1)
ymax=max(samples)
t=fi*16000
i=0
for i in range(len(tryi)-1):
    #print(tr)
    if tryi[i]==0:
        ax2.plot([fi[i],fi[i+1]],[ymax*1.1,ymax*1.1],color='orange')
    elif tryi[i]==1:
        ax2.plot([fi[i],fi[i+1]],[ymax*1.1,ymax*1.1],color='red')
    
#print(i)
if tryi[i]==0:
    ax2.plot([fi[i],duration],[ymax*1.1,ymax*1.1])
else:
    ax2.plot([fi[i],duration],[ymax*1.1,ymax*1.1])
    
plt.savefig('final_results_hac.jpg')

