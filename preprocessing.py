# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:13:44 2020

@author: JARVIS
"""

from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

dataset = 'dataset'

embeddingFile = 'output/embeddings.pickle' # Initial name for embedding pickle
embeddingModel = 'openface.nn4.small2.v1.t7' # Initialising model for embedding pytorch

# Initialization of caffe model for face detection
prototxt = 'model/deploy.prototxt'
model = 'model/res10_300x300_ssd_iter_140000.caffemodel'

# Loading caffe model for face detection and detecting face from image via caffe deep learning
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Loading pytorch model file for extracting facial embeddings via deep learning feature extraction
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Getting image paths
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbedings = []
knownNames = []
total = 0
conf = 0.5

# Read images one-by-one and apply face-detection and embedding
for i, imagePath in enumerate(imagePaths):
    print(f'Processing image {i+1}/{len(imagePaths)}')
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width = 600)
    h, w = image.shape[:2]
    
    # Converting image to blob for dnn face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB = True, crop = True)
    
    # Setting input blob image
    detector.setInput(imageBlob)
    
    # Prediction of the face
    detections = detector.forward()
    
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :,2])
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf:
            # ROI range of interest
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')
            face = image[startY:endY, startX:endX]
            fH, fW = face.shape[:2]
            
            if fH < 20 or fW < 20:
                continue
            
            # Image blob for face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB = True, crop = True)
            
            # Facial feature embedder input image face blob
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownNames.append(name)
            knownEmbedings.append(vec.flatten())
            total += 1
            
print(f'Embedding: {total}')
data = {'embeddings': knownEmbedings, 'names': knownNames}
f = open(embeddingFile, 'wb')
f.write(pickle.dumps(data))
f.close()
print("Process Ended")