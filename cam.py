import cv2
import keras
import detection
import numpy as np

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

# autocoder = keras.models.load_model("/home/maxim/proga/python/face2.0/models/model1/9_91.hdf5")
autocoder = keras.models.load_model("/home/maxim/proga/python/face2.0/facepoints_model.hdf5")

while rval:
    coords = detection.detect1(autocoder, frame)


    for c in coords:
        for x in range(-3, 3):
            for y in range(-3, 3):
                frame[c[0] + y, c[1] + x] = [0, 0, 255]
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()