import os
from PIL import Image
import cv2
import numpy as np



os.chdir(r"C:\Users\Lenovo\Desktop\pythonfolder\project\TrainingImages")
directory=os.getcwd()
print(directory)
# path=os.path
# print(directory,"++++",__file__)
# # # print(type(directory))
# # # print(os.listdir())
# # # print(os.path)




def train_classfier():
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
    print(ids)
    clf=cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classfier.xml")


   


train_classfier()
