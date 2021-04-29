# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:37:50 2019

@author: Anshul
"""

from pyAudioAnalysis.audioSegmentation import speakerDiarization
x,timeL,mt_feats_norm = speakerDiarization('Cropped_5MIN.wav',n_speakers=-1,plot_res=1)

from pyAudioAnalysis import audioSegmentation as aS
[flagsInd, classesAll, acc, CM] = aS.mtFileClassification(input_file='Original_5MIN.wav',model_name='data/svmSM',model_type='svm',plot_results=True)
[flagsInd, classesAll, acc, CM] = aS.mtFileClassification("data/scottish.wav", "data/svmSM", "svm", True, 'data/scottish.segments')

flags_ind=[]

flags_ind.append(res)
import numpy as np
flags_ind = np.array(timeL)
for i in range(1, len(flags_ind) - 1):
        if flags_ind[i-1] == flags_ind[i + 1]:
            flags_ind[i] = flags_ind[i + 1]



np.savetxt('mt_feats_norms.csv',mt_feats_norm,delimiter=',')