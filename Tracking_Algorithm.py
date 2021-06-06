import cv2
import matplotlib.pyplot  as plt
from sklearn.externals import joblib
from skimage import feature
import os
import csv
import time

#function for entering values in csv file
def fun(i):
    with open ('./test data/%s.csv'%(i),'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        now = time.strftime('%d-%m-%Y %H:%M:%S')
        writer.writerow([now])


samples=500

if not os.path.exists('test data'):
    os.makedirs('test data')

a=os.listdir('train data')
for i in a:
    if not os.path.isfile('./test data/%s.csv'%(i)):
        with open('./test data/%s.csv'%(i), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                

vid=cv2.VideoCapture(0)

#using the saved model
model=joblib.load('svm_model_face.pkl')

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

        #roi_gray = im1[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(im_f)
        
        if type(eyes)!=tuple:
            cv2.rectangle(frame,(x,y),(x+w,y+h),[0,255,255],3)
            im_f=cv2.resize(im_f,(112,92))
            f=feature.hog(im_f,transform_sqrt=True,block_norm='L1')
            f=f.reshape(1,-1)
            b=model.predict(f)
            
            bb=a[int(b)]
            fun(bb)
            iter1+=1
            cv2.putText(frame,' '+bb,(x,y),cv2.FONT_ITALIC,1,(225,0,0),2,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    cv2.waitKey(1)
vid.release()
cv2.destroyAllWindows()


