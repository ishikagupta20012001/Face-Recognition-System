import os
from PIL import Image
import cv2
import numpy as np


def capture_image():
    face_cascade = cv2.CascadeClassifier("face_cascade.xml")
    video=cv2.VideoCapture(0)
    check,frame=video.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.03, minNeighbors=5)

    for x,y,w,h in faces:
        gray_img=cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3)
        X,Y=x,y
    cv2.imshow('Capturing video',gray_img)
    key=cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.read(r"C:\Users\Lenovo\Desktop\pythonfolder\project\TrainingImages\classfier.xml")
    id,conf=clf.predict(gray_img[Y:Y+h,X:X+w])
    print(id,conf)




capture_image()











    