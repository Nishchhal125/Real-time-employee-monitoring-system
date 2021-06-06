#Training of data using hog and svm

from sklearn import metrics,datasets,svm
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import pandas as pd
from skimage import feature
from sklearn.externals import joblib
import os


a=os.listdir('train data')
b=os.listdir('./train data/%s'%(a[0]))

dataset=np.zeros((len(a)*len(b),8748))
data_label=np.zeros((len(a)*len(b),))


c=0
for i in range(0,len(a)):
    b=os.listdir('./train data/%s'%(a[i]))
    for j in range(0,len(b)):
        path='./train data/%s/%s'%(a[i],b[i])
        x=mimg.imread(path)
        f=feature.hog(x,transform_sqrt=True,block_norm='L1') #hog to tackle occlusion and better accuracy 
        f=f.reshape(1,-1)
        dataset[c,:]=f
        data_label[c]=i
        c+=1

#SVM classifier
svm_model=svm.SVC(kernel='linear',gamma='auto')

svm_model=svm_model.fit(dataset,data_label)

#Saving the model
joblib.dump(svm_model,'svm_model_face.pkl')

