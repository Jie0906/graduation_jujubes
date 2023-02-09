import os
import spectral
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage import measure, segmentation,color
from sklearn.decomposition import PCA


def test_peanut_roi(img,mask):
    try: 
        peanut_roi = []
        peanut_coordinate = []
        img_la_overlay = []
        cleared = mask.copy()  #複製
        segmentation.clear_border(cleared)
        label_image =measure.label(cleared)  #連續區域標記
        borders = np.logical_xor(mask, cleared) #異物
        label_image[borders] = -1
       
        x = 2
        for region in measure.regionprops(label_image):      
            #忽略小區域
            if region.area <1000:
                continue
            #ROI
            minr, minc, maxr, maxc = region.bbox      
            peanut_roi.append(img[minr-x:maxr+x,minc-x:maxc+x,:])
            peanut_coordinate.append(region.bbox)
            
    except:
        print('ROI error')
    else:
        return peanut_roi
x = []
y = []
data = []
good = []
x = np.zeros(8)
y = np.zeros(8)
res = []


filename = 'C:/Users/hsu/Desktop/g/g/1_New-1'
img = envi.open(filename + '.hdr')
him = img.asarray()
plt.figure()
p = img.read_band(1)
plt.imshow(p)
t = plt.ginput(8)
plt.close()
print(t)
    
for i in range(8):
    (x[i],y[i]) = np.array(t[i])

x = x.astype(int)
y = y.astype(int)

for i in range(8):
    data.append(him[y[i],x[i]])    

clf = SVC(gamma='auto')
b = np.array([0,0,0,0,1,1,1,1])
clf.fit(data,b)
pre = clf.predict(him.reshape(img.shape[0]*img.shape[1],img.shape[2]))


for i in range(25):

    filename = f'C:/Users/hsu/Desktop/g/g/{i+1}_New-1'
    img = envi.open(filename + '.hdr')
    him = img.asarray()
    pre = pre.reshape(img.shape[0],img.shape[1])
    # plt.figure()
    # plt.imshow(pre)
    roi = test_peanut_roi(him,pre)
    
    for i in range(len(roi)):
        roi_resize = cv2.resize(roi[i], (200,200) )
        reshape_roi = np.reshape(roi_resize,(roi_resize.shape[0]*roi_resize.shape[1],roi_resize.shape[2]))
        pca = PCA(n_components=3)
        pca.fit(reshape_roi)
        new_pca = pca.transform(reshape_roi)
        new_img = np.reshape(new_pca,(roi_resize.shape[0],roi_resize.shape[1],-1))
        res.append(new_img)
        good.append(1)
res = np.array(res)
np.save('good_result.npy',res)

'''
for i in range(25):
    filename = f'C:/Users/hsu/Desktop/g/g/{i+1}_New-1'
    img = envi.open(filename + '.hdr')
    him = img.asarray()
    pre = clf.predict(him.reshape(img.shape[0]*img.shape[1],img.shape[2]))
    pre = pre.reshape(img.shape[0],img.shape[1])
    plt.figure()
    plt.imshow(pre)
    roi = test_peanut_roi(him,pre)

    for i in range(4):
        res.append(roi)

      
np.savez('result.npz',res)

'''

'''
for i in range(25):
    filename = f'C:/Users/hsu/Desktop/g/g/{i+1}_New-1'
    img = envi.open(filename + '.hdr').asarray()

plt.imshow(img)
    
'''
'''
for i in range(0,img.shape[2]):
    p = img.read_band(i)
    plt.imshow(p)
    plt.show()

'''

