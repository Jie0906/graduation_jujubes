import os
import cv2
import pickle
import spectral
import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D, GlobalAveragePooling3D,Conv1D, MaxPool1D, GlobalAveragePooling1D, Conv2DTranspose, Conv3DTranspose,BatchNormalization 
from keras.models import Sequential, load_model, Model 
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import time
import datetime

#%%
def calc_R(a):

    r = np.transpose(np.reshape(a, [-1, a.shape[2]]))
    R = 1/(a.shape[0]*a.shape[1])*np.dot(r, np.transpose(r))        
    return R

def calc_K_u(HIM):
    try:
        N = HIM.shape[0]*HIM.shape[1]
        r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
        u = (np.mean(r, 1)).reshape(HIM.shape[2], 1)        
        K = 1/N*np.dot(r-u, np.transpose(r-u))
        return K, u
    except:
        print('An error occurred in calc_K_u()')
        
        
        
def tcimf(HIM, d, no_d):
    '''
    Target-Constrained Interference-Minimized Filter
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    param no_d: undesired target, type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    result = np.zeros([HIM.shape[0]*HIM.shape[1], 1])
    DU = np.hstack(([d, no_d]))
    d_count = d.shape[1]
    no_d_count = no_d.shape[1]
    DUtw = np.zeros([d_count + no_d_count, 1])
    DUtw[0: d_count] = 1
    R = (1/HIM.shape[0]*HIM.shape[1])*np.dot(r, np.transpose(r))
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
    x = np.dot(np.dot(np.transpose(r), Rinv), DU)
    y = np.dot(np.dot(np.transpose(DU), Rinv), DU)
    y = np.linalg.inv(y)
    result = np.dot(np.dot(x, y), DUtw)  
    result = np.reshape(result, HIM.shape[:-1])
    return result


def osp(HIM, d, no_d):
    '''
    Orthogonal Subspace Projection
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param no_d: undesired target, type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    I = np.eye(HIM.shape[2]) 
    P = I - (no_d@np.linalg.inv( (no_d.T)@no_d ))@(no_d.T)
    x = (d.T)@P@r
    result = np.reshape(x, HIM.shape[:-1])
    return result
        
def hcem(HIM, d, max_it = 100, λ = 200, e = 1e-6):
    '''
    Hierarchical Constrained Energy Minimization
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param max_it: maximum number of iterations, type is int
    param λ: coefficients for constructing a new CEM detector, type is int
    param e: stop iterating until the error is less than e, type is int
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    D, N = r.shape  # bands, pixel number
    hCEMMap_old = np.ones([1, N])
    Weight = np.ones([1, N])
    
    for i in range(max_it):
        r = r*Weight
        R = 1/N*(r@r.T)
        Rinv = np.linalg.inv(R + 0.0001*np.eye(D))
        w = (Rinv@d) / (d.T@Rinv@d)
        hCEMMap = w.T@r
        
        Weight = 1 - np.exp(-λ*hCEMMap)
        Weight[Weight < 0] = 0
        
        res = np.power(np.linalg.norm(hCEMMap_old), 2)/N - np.power(np.linalg.norm(hCEMMap), 2)/N
        print(f'iteration {i+1}: ε = {res}')
        hCEMMap_old = hCEMMap.copy()
        if np.abs(res) < e:
            break
    hCEMMap = hCEMMap.reshape(HIM.shape[:-1])
    return hCEMMap

def snr(HIM, d, R = None):
    R = calc_R(HIM)
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    D, N = r.shape
    Rinv = np.linalg.inv(R)
    result = np.dot(np.transpose(r), np.dot(Rinv, d))
    return result

def nlrt(HIM, d, k = None):
    if k is None :
        k, u = calc_K_u(HIM)
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    D, N = r.shape
    Rinv = np.linalg.inv(k)
    result = np.dot(np.transpose(r), np.dot(Rinv, d))/np.dot(np.transpose(d), np.dot(Rinv, d))

    return result

def lrt(HIM, d, k = None):
    if k is None :
        k, u = calc_K_u(HIM)
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    D, N = r.shape
    Rinv = np.linalg.inv(k)
    result = np.dot(np.transpose(r), np.dot(Rinv, d))

    return result

def amd(HIM, d, K = None, u = None ):

    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if K is None or u is None:
        K, u = calc_K_u(HIM)
        Kinv = np.linalg.inv(K)

        
    result = (d-u).T@Kinv@(r-u)  
    result = np.reshape(result, HIM.shape[:-1])
    return result

def namd(HIM,d,K=None,u=None):
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if K is None:
       K, u = calc_K_u(HIM)
    ru = r-u
    du = d-u
    
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
    result = (du.T@Kinv@ru)/(du.T@Kinv@du)
    result = np.reshape(result, HIM.shape[:-1])
    return result

def cem(apple,d_point):
    
    ri = apple.reshape(x * y,z)
    n = x * y
    R = np.dot(np.transpose(ri),ri) / n
    R_inv = np.linalg.inv(R)

        
    ans = (np.dot(ri, np.dot(R_inv, d_point))) / (np.dot(np.transpose(d_point), np.dot(R_inv, d_point)))
    
    CEM_result = ans.reshape(x, y)
    
    return CEM_result


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

        
def Patch(data,height_index,width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    
    return patch


#%% filters

dates = spectral.envi.open('dates_complex1_RT.hdr').asarray()
#gt = scipy.io.loadmat('dates_gt.mat') 

xx = [20,60,100]
false_color_bands = np.array(xx)

img = dates[..., false_color_bands]

plt.figure()
# plt.imshow(dates[:,:,133])
plt.imshow(img)

#%%
dates = spectral.envi.open('dates_complex1_RT.hdr').asarray()
gt = scipy.io.loadmat('dates_complex_gt') 


cem_result = []
namd_result = []
amd_result = []
lrt_result = []
nlrt_result = []
snr_result = []
hcem_result = []
x, y, z = dates.shape
#get ground_truth
plt.figure()
plt.imshow(gt['dates'])
d_point = plt.ginput(2, timeout = 300)

for i in range(len(d_point)):
    d_point[i] = dates[int(d_point[i][1]),int(d_point[i][0])]
plt.close()

for i in range(len(d_point)):
    cem_temp = cem(dates,d_point[i])
    namd_temp = namd(dates,d_point[i])
    amd_temp = amd(dates,d_point[i])
    lrt_temp = lrt(dates,d_point[i])
    nlrt_temp = nlrt(dates,d_point[i])
    snr_temp = snr(dates,d_point[i])

   
    print('processed  ' + str(i) + ' point')
   
    cem_result.append(cem_temp)
    namd_result.append(namd_temp)
    amd_result.append(amd_temp)
    lrt_result.append(lrt_temp)
    nlrt_result.append(nlrt_temp)
    snr_result.append(snr_temp)

   
cem_result = np.stack((cem_result[0],cem_result[1]),axis=-1) 
namd_result = np.stack((namd_result[0],namd_result[1]),axis=-1)    
amd_result = np.stack((amd_result[0],amd_result[1]),axis=-1) 
lrt_result = np.stack((lrt_result[0],lrt_result[1]),axis=-1) 
nlrt_result = np.stack((nlrt_result[0],nlrt_result[1]),axis=-1) 
snr_result = np.stack((snr_result[0],snr_result[1]),axis=-1)    


np.save('dates_cem_result.npy', cem_result)
np.save('dates_namd_result.npy', namd_result)
np.save('dates_amd_result.npy', amd_result)
np.save('dates_lrt_result.npy', lrt_result)
np.save('dates_nlrt_result.npy', nlrt_result)
np.save('dates_snr_result.npy', snr_result)


#%%處理2D
dates = spectral.envi.open('dates_complex1_RT.hdr').asarray()
gt = scipy.io.loadmat('dates_complex_gt') 

dates_cem_result = np.load('dates_cem_result.npy',allow_pickle=True)
dates_namd_result = np.load('dates_namd_result.npy',allow_pickle=True)
dates_amd_result = np.load('dates_amd_result.npy',allow_pickle=True)
dates_lrt_result = np.load('dates_lrt_result.npy',allow_pickle=True)
dates_nlrt_result = np.load('dates_nlrt_result.npy',allow_pickle=True)
dates_snr_result = np.load('dates_snr_result.npy',allow_pickle=True)


windowSize = 35

dates_cem_result = np.reshape(dates_cem_result, (1052, 1024 ,2))
dates_namd_result = np.reshape(dates_namd_result, (1052, 1024 ,2))
dates_amd_result = np.reshape(dates_amd_result, (1052, 1024 ,2))
dates_lrt_result = np.reshape(dates_lrt_result, (1052, 1024 ,2))
dates_nlrt_result = np.reshape(dates_nlrt_result, (1052, 1024 ,2))
dates_snr_result = np.reshape(dates_snr_result, (1052, 1024 ,2))


dates_cem_result_slice, gt_slice = createImageCubes(dates_cem_result, gt['dates'], windowSize=windowSize)
dates_namd_result_slice, gt_slice = createImageCubes(dates_namd_result, gt['dates'], windowSize=windowSize)
dates_amd_result_slice, gt_slice = createImageCubes(dates_amd_result, gt['dates'], windowSize=windowSize)
dates_lrt_result_slice, gt_slice = createImageCubes(dates_lrt_result, gt['dates'], windowSize=windowSize)
dates_nlrt_result_slice, gt_slice = createImageCubes(dates_nlrt_result, gt['dates'], windowSize=windowSize)
dates_snr_result_slice, gt_slice = createImageCubes(dates_snr_result, gt['dates'], windowSize=windowSize)


np.save('gt.npy',gt['dates'])
np.save('dates_cem_result_slice.npy',dates_cem_result_slice)
np.save('dates_namd_result_slice.npy',dates_namd_result_slice)
np.save('dates_amd_result_slice.npy',dates_amd_result_slice)
np.save('dates_lrt_result_slice.npy',dates_lrt_result_slice)
np.save('dates_nlrt_result_slice.npy',dates_nlrt_result_slice)
np.save('dates_snr_result_slice.npy',dates_snr_result_slice)

#%% 2D CNN

def cov2D(cnn_filter, counting_run):
    acc_list  = []
    acc_list_top5 = []
    origin_train, origin_test, ans_train, ans_test = train_test_split( cnn_filter, gt_slice, test_size = 0.85, random_state = 0)
    ans_train = np_utils.to_categorical(ans_train)
    ans_test = np_utils.to_categorical(ans_test)
    
    for i in range(counting_run):
        
        
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=3, input_shape=(35, 35, 2), activation='relu', padding = 'same'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding = 'same'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())  
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # filepath="model_{epoch:02d}-{val_accuracy:.2f}.h5"
        # checkpoint = ModelCheckpoint(
        #     filepath = filepath,
        #     monitor='val_acc',
        #     save_best_only=True,
        #     mode='max')
        
        # callbacks_list = [checkpoint]
        
        start = time.time()
        result = model.fit(origin_train, ans_train, epochs=100, batch_size= 100, validation_split = 0.05, verbose=1)
        print('result: ',result)
        
        
        print("Total training time: ", time.time() - start, "seconds")
        
        
        loss, accuracy = model.evaluate(origin_test, ans_test)
        print('Test:')
        print('Loss:', loss)
        print('Accuracy:', accuracy)
        acc_list.append(accuracy)
        acc_list.sort(reverse=True)
        acc_list_top5 = acc_list[0:5]
       

    sum = 0
    for i in range(len(acc_list_top5)):
        sum+= acc_list_top5[i]
    ave = sum/len(acc_list_top5)
    
    print("accurary: "+ str(acc_list))
    print("average: "+ str(ave))
    
    #save model
    model.save("dates_complex_namd_model.h5")
    
    
    #predict
    
    dataset = "dates"
    # load best weights
    model.load_weights("dates_complex_namd_model.h5")
    # model_combined.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Xtest_2D = Xtest_2D.reshape(-1, windowSize, windowSize, 16)
    # Xtest_3D = Xtest_3D.reshape(-1, windowSize, windowSize, 30)
    # ytest = np_utils.to_categorical(ytest)
    Y_pred_test = model.predict(origin_test)
    y_pred_test = np.argmax(Y_pred_test, axis=1)

    classification = classification_report(np.argmax(ans_test, axis=1), y_pred_test)
    print(classification)

    def AA_andEachClassAccuracy(confusion_matrix):
        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def reports (origin_test, ans_test, name):
        start = time.time()
        Y_pred = model.predict(origin_test)
        y_pred = np.argmax(Y_pred, axis=1)
        end = time.time()
        print('pridict_time: ',end - start)
        if name == 'IP':
            target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                            ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                            'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                            'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                            'Stone-Steel-Towers']
        elif name == 'SA':
            target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                            'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                            'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                            'Vinyard_untrained','Vinyard_vertical_trellis']
        elif name == 'PU':
            target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                            'Self-Blocking Bricks','Shadows']
        elif name == 'peanut':
            target_names = ['good','bad']
        elif name == 'dates':
            target_names = ['good','bad']
        
        classification = classification_report(np.argmax(ans_test, axis=1), y_pred, target_names=target_names)
        oa = accuracy_score(np.argmax(ans_test, axis=1), y_pred)
        confusion = confusion_matrix(np.argmax(ans_test, axis=1), y_pred)
        each_acc, aa = AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(np.argmax(ans_test, axis=1), y_pred)
        score = model.evaluate(origin_test, ans_test, batch_size=100)
        Test_Loss =  score[0]*100
        Test_accuracy = score[1]*100
        return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100


    classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(origin_test, ans_test, dataset)
    classification = str(classification)
    confusion = str(confusion)
    file_name = "dates_complex_namd_detect.txt"

    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

#load the original image
    # X= dates_namd_result
    # y = gt['dates']
    # height = y.shape[0]
    # width = y.shape[1]
    # PATCH_SIZE = windowSize
    # X = padWithZeros(X, PATCH_SIZE//2)
    # # calculate the predicted image
    # outputs = np.zeros((height,width))
    # for i in range(height):
    #     for j in range(width):
    #         target = int(y[i,j])
    #         if target == 0 :
    #             continue
    #         else :
    #             image_patch=Patch(X,i,j, PATCH_SIZE )
    #             X_test_image = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1).astype('float32')                                   
    #             prediction = (model.predict(X_test_image))
    #             prediction = np.argmax(prediction, axis=1)
    #             outputs[i][j] = prediction+1
    # predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))
    # spectral.save_rgb("predictions.jpg", outputs.astype(int), colors=spectral.spy_colors)

    


#%% main
cov2D(dates_namd_result_slice, 1)

