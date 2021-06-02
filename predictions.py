# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:16:20 2020

@author: JARVIS
"""

# Importing libraries
import numpy as np
import imutils
import pickle
import time
import cv2

embeddingModel = 'openface.nn4.small2.v1.t7'
embeddingFile = 'output/embeddings.pickle'
recognizerFile = 'output/recognizer.pickle'
labelEncFile = 'output/labelEncoder.pickle'
conf = 0.5

print('Loading face detector...')
prototxt = 'model/deploy.prototxt'
model = 'model/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print('Loading face recognizer...')
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, 'rb').read())
le = pickle.loads(open(labelEncFile, 'rb').read())

box = []
print('Starting video stream...')
cam = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width = 600)
    h, w = frame.shape[:2]
    # Converting image to blob for dnn face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB = True, crop = True)
    
    # Setting input blob image
    detector.setInput(imageBlob)
    
    # Prediction of the face
    detections = detector.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')
            face = frame[startY:endY, startX:endX]
            fH, fW = face.shape[:2]
            
            if fH < 20 or fW < 20:
                continue
            
            # Image blob for face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB = True, crop = True)
            
            # Facial feature embedder input image face blob
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            prob = preds[j]
            name = le.classes_[j]
            text = f'{name}: {prob*100}%'
            y = startY-10 if startY- 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0))

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()