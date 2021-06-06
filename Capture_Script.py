#To create folders and store images of every person
import cv2
from skimage import feature
import matplotlib.pyplot as plt
import os

if not os.path.exists('train data'):
    os.makedirs('train data')

no=input('enter the no. of people to be trained')

for i in range (0,int(no)):
    name=input('enter name to be trained ')

    if not os.path.exists('./train data/%s'%(name)):
        os.makedirs('./train data/%s'%(name))
    else:
        print('Invalid name:already exits')
        name=input('enter again')


    samples=20
    vid=cv2.VideoCapture(0)

    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml')
    iter1=0
    while(iter1<samples):
        r,frame=vid.read()
        frame=cv2.resize(frame,(640,480))
        im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        face=face_cascade.detectMultiScale(im1,1.05,6,minSize=(50,50))
        for x,y,w,h in (face):

            im_f=im1[y:y+h,x:x+w]
            roi_gray = im1[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)#to check for eyes in the face

            #condition used for minimal false detection
            
            if type(eyes)!=tuple:  
                cv2.rectangle(frame,(x,y),(x+w,y+h),[0,255,255],3)
                im_f=cv2.resize(im_f,(112,92))
                cv2.putText(frame,'face no.'+str(iter1),(x,y),cv2.FONT_ITALIC,1,(225,0,0),2,cv2.LINE_AA)
                
                iter1+=1
                path='./train data/%s/%d.png'%(name,iter1)

                cv2.imwrite(path,im_f)
                


        cv2.imshow('frame',frame)
        cv2.waitKey(100)
    vid.release()
    cv2.destroyAllWindows()
    
        
