import rawpy
import imageio
import matplotlib.pylab as plt
import os
import matplotlib.pyplot as plt
import spectral
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import cv2
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import *
import pandas as pd
from keras.models import Sequential, load_model, Model
from keras.utils import np_utils
import tensorflow as tf
from sklearn.decomposition import PCA
from PIL import Image
import scipy.io

#%% often function


def save_jpg(img, file_name ):
    height, width ,b= img.shape
    fig = plt.figure(file_name)
    plt.axis('off')
    fig.set_size_inches(width/100.0,height/100.0)  #輸出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0, wspace=0)
    plt.margins(0,0)
    plt.imshow(img)
    plt.show()
    plt.savefig(f'{file_name}.jpg', pad_inches=0.0)
    
def save_npy(img, file_namme ):
    np.save(f'{file_namme}.npy', img)

#%% show false_color band

dates = spectral.envi.open('dates_complex3_RT.hdr').asarray()
#gt = scipy.io.loadmat('dates_gt.mat') 

xx = [10,40,60]
false_color_bands = np.array(xx)

img = dates[..., false_color_bands]

plt.figure()
plt.imshow(img)



#%% raw to rgb

iamge = '1_New-1'
data = spectral.envi.open(iamge+'.hdr').asarray()
file = data[...,20:140]
xx = [8,16,24]
false_color_bands = np.array(xx)

img = data[..., false_color_bands]

save_jpg(img)


#%%　dates raw to rgb (size = 1052*1024*224)
dates = spectral.envi.open('dates_complex1_RT.hdr').asarray()
#gt = scipy.io.loadmat('peanut_gt.mat') 

select_bnands = [20,60,100]
false_color_bands = np.array(select_bnands)

img = dates[..., false_color_bands]

#plt.figure()
# plt.imshow(leather[:,:,133])
#plt.imshow(img)
save_jpg(img, 'dates_complex')


#%% show rgb

root = r'C:\Users\user\Desktop\昱杰\蜜棗檔案\0204蜜棗\1+2級\dates_complex_printer'


img = plt.imread(root + '.jpg')
plt.Figure()
plt.imshow(img)


#%% remove_background
root = r'C:\Users\user\Desktop\CNN\Desktop\CNN\peanut'
pick_point = []

gt = plt.imread(root + '.jpg')
plt.Figure()
plt.imshow(gt)

pick_ans = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
 
pick_temp = plt.ginput(15,show_clicks=True,timeout = 300)
plt.show()

for i in range(len(pick_temp)):
    pick_point.append(gt[int(pick_temp[i][1]),int(pick_temp[i][0]),:])
    
plt.close()

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
result = clf.fit(pick_point, pick_ans)  

mask = clf.predict(np.reshape(gt,(216*409,3)))


#mask=clf.predict(origin.reshape(origin[0]*origin[1],25))
mask=mask.reshape(216,409)
plt.figure() 
plt.imshow(mask)


#%% process after printer(dates)

root = r'C:\Users\user\Desktop\昱杰\蜜棗檔案\0204蜜棗\1+2級\dates_complex_printer'

img = plt.imread(root + '.jpg')
plt.imshow(img)
classfication = np.zeros(img.shape[0]*img.shape[1])
img_gt = img.reshape([-1, 3])
img_gt_test = img.reshape([img.shape[0] * img.shape[1],img.shape[2]])
classfication = np.zeros(img_gt.shape[0])

for i in range(img_gt.shape[0]):
    if img_gt[i, 0] > 200 and img_gt[i, 1] < 200 and img_gt[i, 2]< 200:
        classfication[i] = 1
    if img_gt[i, 0] < 200 and img_gt[i, 1] < 200 and img_gt[i, 2]> 200:
        classfication[i] = 2
        
temp = np.reshape(classfication, (1052,1024))
        
plt.figure()
plt.imshow(temp)

save_jpg(temp, 'dates_complex_gt')
# scipy.io.savemat('dates_complex_gt.mat',{'dates':temp})
# save_npy(temp, 'dates_complex_gt')

#%% provess after printer (leather)

root = r'C:\Users\user\Desktop\0811\leather_printer'

img = plt.imread(root + '.jpg')
plt.imshow(img)
classfication = np.zeros(img.shape[0]*img.shape[1])
img_gt = img.reshape([-1, 3])
img_gt_test = img.reshape([img.shape[0] * img.shape[1],img.shape[2]])
classfication = np.zeros(img_gt.shape[0])

for i in range(img_gt.shape[0]):
    if img_gt[i, 0] > 200 and img_gt[i, 1] < 200 and img_gt[i, 2]< 200:
        classfication[i] = 1
    else:
        classfication[i] = 2

        
temp = np.reshape(classfication, (1088,2048))
        
plt.figure()
plt.imshow(temp)

scipy.io.savemat('leather_gt.mat',{'leather':temp})
save_npy(temp, 'leather_gt')