# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 08:48:29 2021

@author: JARVIS
"""

# Import necessary libraries
import numpy as np
import imutils
import pickle
import cv2
import time
from gtts import gTTS
from playsound import playsound

language = 'en'

# Initialize file names of object detection model
prototxt_obj_detect = 'MobileNetSSD_deploy.prototxt.txt'
model_obj_detect = 'MobileNetSSD_deploy.caffemodel'
confThresh = 0.2

# Initialize list of classes detected by object detection model
CLASSES = ['background', 'aerospace', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'notorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# Generates random colors to be used while drawing boxes on detected onbjects
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize file names of face recognition model
embeddingModel = 'openface.nn4.small2.v1.t7'
embeddingFile = 'output/embeddings.pickle'
recognizerFile = 'output/recognizer.pickle'
labelEncFile = 'output/labelEncoder.pickle'
conf = 0.5

# Load face detector
print('Loading face detector...')
prototxt_face_recog = 'model/deploy.prototxt'
model_face_recog = 'model/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(prototxt_face_recog, model_face_recog)

# Load face recognizer
print('Loading face recognizer...')
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, 'rb').read())
le = pickle.loads(open(labelEncFile, 'rb').read())

# Initialize empty list for storing (startx, starty) and (endx, endy) coordinates of the detected objects
box = []

# Load object detection model
print('loading object detection model...')
net = cv2.dnn.readNetFromCaffe(prototxt_obj_detect, model_obj_detect)
print('model loaded!\nstarting camera feed...')

# Streaming from camera
cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()             # capture frame
    # resize frame(only resized so that it fits on display while displaying output)
    frame = imutils.resize(frame, width=500)
    h, w = frame.shape[:2]              # get height and width of the frame
    # resize frame to 300x300(because both object detection and face recognition models work with 300x300 images)
    imResizeBlob = cv2.resize(frame, dsize=(300, 300))
    # create blob of the image(models are traind using blob)
    blob = cv2.dnn.blobFromImage(imResizeBlob, 0.007843, (300, 300), 127.5)
    net.setInput(blob)                  # set input blob image
    detections = net.forward()          # detect the objects
    detShape = detections.shape[2]      # get dimensions of detections

    # iterate throgh all the detected objects
    for i in np.arange(0, detShape):
        # grab confidence value of the detections
        confidence = detections[0, 0, i, 2]
        if confidence > confThresh:
            # if the confidence value of detections is greater than initialized confidence value then do the following
            # gab ID of the detected object(ID is equivalent to index of the name of the object from CLASSES list)
            idx = int(detections[0, 0, i, 1])
            # grab dimensions of the box to be drawn
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            # convert the dimensions to integer values
            startX, startY, endX, endY = box.astype('int')
            # draw rectangle around the object
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), COLORS[idx], 2)
            # checks if starty is at the top edge of the screen
            if startY - 15 > 15:
                y = startY - 15
            else:
                y = startY + 15
            # put text of the detected object and its confidence value on the frame
            cv2.putText(frame,
                        f'{CLASSES[idx]} :- {confidence*100}%',
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2)
            # convert the output and save it to the audio file
            myobj = gTTS(text=CLASSES[idx], lang=language, slow=False)
            myobj.save("detection.mp3")
            # Playing the converted file
            playsound("detection.mp3")
            if CLASSES[idx] == 'person':
                # if the detected CLASS is person then go for face recognition
                # resize original frame (only for display puroposes)
                frame = imutils.resize(frame, width=600)
                h, w = frame.shape[:2]  # grab width and height of the frame
                # Converting image to blob for dnn face detection
                imageBlob = cv2.dnn.blobFromImage(cv2.resize(
                    frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=True, crop=True)

                # Setting input blob image
                detector.setInput(imageBlob)

                # Prediction of the face
                detections = detector.forward()

                # iterate throgh all the detected faces
                for i in range(0, detections.shape[2]):
                    # grab confidence value
                    confidence = detections[0, 0, i, 2]

                    if confidence > conf:
                        # if confidence value of detected face is greater than the initialized value, do the following
                        # grab coordinates of the detected objects
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        # convert the coordinates to integer value
                        startX, startY, endX, endY = box.astype('int')
                        # grab detected face from the frame
                        face = frame[startY:endY, startX:endX]
                        # grab height and width of the face
                        fH, fW = face.shape[:2]

                        if fH < 20 or fW < 20:
                            # if there are multiple faces then start over
                            continue

                        # Image blob for face
                        faceBlob = cv2.dnn.blobFromImage(
                            face, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=True)

                        # Facial feature embedder input image face blob
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        preds = recognizer.predict_proba(
                            vec)[0]  # predict the faces
                        j = np.argmax(preds)  # grab the maximum prediction
                        # grab the probability of the prediction
                        prob = preds[j]
                        name = le.classes_[j]  # get the name
                        text = f'{name}: {prob*100}%'
                        # check if detection is at the edge of the screen
                        y = startY-10 if startY - 10 > 10 else startY + 10
                        # draw rectangle around thw face
                        cv2.rectangle(frame, (startX, startY),
                                      (endX, endY), (0, 0, 255), 2)
                        # display name and confidance value above the detected face
                        cv2.putText(frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0))
                        myobj_name = gTTS(text=name, lang=language, slow=False)
                        myobj_name.save("person.mp3")
                        # Playing the converted file
                        playsound("person.mp3")
    cv2.imshow('Vision', frame)  # display the frame
    key = cv2.waitKey(1)  # wait for press of a key
    if key == 27:
        # if Esc key is pressed, break out of the loop
        break

# turn off the camera and destroy any windows opened by the program
cap.release()
cv2.destroyAllWindows()
