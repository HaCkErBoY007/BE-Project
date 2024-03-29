import numpy as np
import imutils
import cv2
import time

prototxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
confThresh = 0.2

CLASSES = ['background', 'aerospace', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'notorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

print('loading model...')
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print('model loaded!\nstarting camera feed...')

cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 500)
    h, w = frame.shape[:2]
    imResizeBlob = cv2.resize(frame, dsize = (300, 300))
    blob = cv2.dnn.blobFromImage(imResizeBlob, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    detShape = detections.shape[2]
    
    for i in np.arange(0, detShape):
        confidence = detections[0, 0, i, 2]
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype('int')
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            if startY - 15 > 15:
                y = startY - 15
            else:
                y = startY + 15
            cv2.putText(frame,
                        f'{CLASSES[idx]} :- {confidence*100}%',
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2)
    cv2.imshow('ObjectDetection', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()