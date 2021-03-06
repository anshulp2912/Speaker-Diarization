
# <img src="icon.png" align="left" height="130"/> A Python library for audio feature extraction, classification, segmentation and applications

*This doc contains general info. Click [here](https://github.com/tyiannak/pyAudioAnalysis/wiki) for the complete wiki*

## News
 * Checkout the tutorial library for the course ["Multimodal Information Processing & Analysis" of the MSc in Data Science in NCSR Demokritos](https://github.com/tyiannak/multimodalAnalysis)
 * [PLOS-One Paper regarding pyAudioAnalysis](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610) *(please cite!)*
 * Check out [pyVisualizeMp3Tags](https://github.com/tyiannak/pyVisualizeMp3Tags) a python script for visualization of mp3 tags and lyrics
 * Check out [paura](https://github.com/tyiannak/paura) a python script for realtime recording and analysis of audio data
  * Latest pyAudioAnalysis update [2018-08-12] now compatible with Python 3


## General
pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks. Through pyAudioAnalysis you can:
 * Extract audio *features* and representations (e.g. mfccs, spectrogram, chromagram)
 * *Classify* unknown sounds
 * *Train*, parameter tune and *evaluate* classifiers of audio segments
 * *Detect* audio events and exclude silence periods from long recordings
 * Perform *supervised segmentation* (joint segmentation - classification)
 * Perform *unsupervised segmentation* (e.g. speaker diarization)
 * Extract audio *thumbnails*
 * Train and use *audio regression* models (example application: emotion recognition)
 * Apply dimensionality reduction to *visualize* audio data and content similarities

## Installation
 * Clone the source of this library:
 ```
git clone https://github.com/tyiannak/pyAudioAnalysis.git
```
 * Install dependencies:
 ```
pip install -r ./requirements.txt
```
 * Install using pip:
 ```
pip install -e .
```
(also works with pip3 now)

## An audio classification example
> More examples and detailed tutorials can be found [at the wiki](https://github.com/tyiannak/pyAudioAnalysis/wiki)

pyAudioAnalysis provides easy-to-call wrappers to execute audio analysis tasks. Eg, this code first trains an audio segment classifier, given a set of WAV files stored in folders (each folder representing a different class) and then the trained classifier is used to classify an unknown audio WAV file

```
from pyAudioAnalysis import audioTrainTest as aT
aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")
Result:
(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])
```

In addition, command-line support is provided for all functionalities. E.g. the following command extracts the spectrogram of an audio signal stored in a WAV file: `python audioAnalysis.py fileSpectrogram -i data/doremi.wav`

## Further reading
Apart from the current README file and [the wiki](https://github.com/tyiannak/pyAudioAnalysis/wiki), a more general and theoretic description of the adopted methods (along with several experiments on particular use-cases) is presented [in this publication](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610). Please use the following citation when citing pyAudioAnalysis in your research work:
```
@article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}
```

Finally, for Matlab-related audio analysis material check  [this book](http://www.amazon.com/Introduction-Audio-Analysis-MATLAB%C2%AE-Approach/dp/0080993885).

## Author
<img src="https://tyiannak.github.io/files/3.JPG" align="left" height="100"/>

[Theodoros Giannakopoulos](https://tyiannak.github.io),
Director of Machine Learning at [Behavioral Signals](https://behavioralsignals.com)
