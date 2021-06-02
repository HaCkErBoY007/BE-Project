# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:02:28 2020

@author: JARVIS
"""

# Importing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Initializing of embedding and recognizer
embeddingFile = 'output/embeddings.pickle'
#new and empty at initial
recognizerFile = 'output/recognizer.pickle'
labelEncFile = 'output/labelEncoder.pickle'

print('Loading face embeddings...')
data = pickle.loads(open(embeddingFile, 'rb').read())

print('Encoding labels...')
labelEnc = LabelEncoder()
labels = labelEnc.fit_transform(data['names'])

print('Training Model...')
recognizer = SVC(C = 1.0, kernel = 'linear', probability = True)
recognizer.fit(data['embeddings'], labels)

f = open(recognizerFile, 'wb')
f.write(pickle.dumps(recognizer))
f.close()

f = open(labelEncFile, 'wb')
f.write(pickle.dumps(labelEnc))
f.close()