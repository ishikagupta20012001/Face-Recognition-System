import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np


#Creating a graphical user interface
st.title('Face Recognition System')
menu=['Aministrative Panel','Old User','New User']
choice=st.sidebar.selectbox("Menu",menu)


def new_details():
    global submit_button
    global firstname
    global lastname
    global id


    st.subheader('New User Login')
    with st.form(key='form'):
        firstname=st.text_input("Enter your first name")

        lastname=st.text_input("Enter your last name")
        id=st.text_input("Enter your id")
        submit_button=st.form_submit_button(label='Login')



def Takeimages():
    face_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\pythonfolder\project\face_cascade.xml")
    video=cv2.VideoCapture(0)
    a=0
    while True:
        check,frame=video.read()
        cv2.imshow('Capturing video',frame)
        gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.03, minNeighbors=5)

        
        for x,y,w,h in faces:
            gray_img=cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3)
            a=a+1
            cv2.imwrite(
                        "TrainingImages\ "+firstname +"."+id +'.'+ str(
                            a) + ".jpg", gray_img[y:y + h, x:x + w])
            

        key=cv2.waitKey(1)
        if key==ord('q'):
            break
        elif a>60:
            break
    video.release()
    cv2.destroyAllWindows()



def change_directory():
    global directory
    os.chdir(r"C:\Users\Lenovo\Desktop\pythonfolder\project\TrainingImages")
    directory=os.getcwd()
    train_classfier()

# print(directory)



def train_classfier():
    os.remove('classfier.xml')
    path=[os.path.join(directory,f)for f in os.listdir(directory)]
    faces=[]
    ids=[]
    for image in path:
        img=Image.open(image).convert('L')
        imagenp=np.array(img,'uint8')
        id=image.split(directory)[1].split('.')[1]
        ids.append(int(id))
        faces.append(imagenp)

    ids=np.array(ids)
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classfier.xml")




def capture_image():
    face_cascade = cv2.CascadeClassifier("face_cascade.xml")
    video=cv2.VideoCapture(0)
    check,frame=video.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.03, minNeighbors=5)
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.read(r"C:\Users\Lenovo\Desktop\pythonfolder\project\TrainingImages\classfier.xml")

    for x,y,w,h in faces:
        gray_img=cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),3)
        id,conf=clf.predict(gray_img[y:y+h,x:x+w])
    cv2.imshow('Capturing video',gray_img)
    key=cv2.waitKey(0)
    video.release()
    cv2.destroyAllWindows()
    st.header(id)   




if(choice =='New User'):
    # os.remove(r'TrainingImages\classfier.xml')
    new_details()
    if submit_button:
        Takeimages()
        change_directory()
elif(choice=='Old User'):
    capture_image()













