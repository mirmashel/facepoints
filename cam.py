import numpy as np
import cv2
import keras
import detection
import skimage

faceCascade = cv2.CascadeClassifier('/home/maxim/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
autocoder = keras.models.load_model("/home/maxim/proga/python/face/facepoints_model.hdf5")

cap = cv2.VideoCapture(0)
cap.set(3, 1080) # set Width
cap.set(4, 960) # set Height


while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        coords = detection.detect1(autocoder, img[y:y+h, x:x+h])
        for c in coords:
            for x1 in range(-3, 3):
                for y1 in range(-3, 3):
                    img[y + c[0] + y1, x + c[1] + x1] = [0, 0, 255]

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]  

    # large = skimage.transform.resize(img, (960, 1080))
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
