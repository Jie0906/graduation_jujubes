import os
import matplotlib.pyplot as plt
import spectral
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow.keras.layers import Conv3D,Conv2D,MaxPool2D, MaxPooling3D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D, GlobalAveragePooling3D,Conv1D, MaxPool1D, GlobalAveragePooling1D, Conv2DTranspose, Conv3DTranspose,BatchNormalization 
import cv2
from peanut_class import peanut
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import *
import pandas as pd
from keras.models import Sequential, load_model, Model
from keras.utils import np_utils
import tensorflow as tf
from sklearn.decomposition import PCA
import time
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

#%%

date_good1 = spectral.envi.open('dates1_1_RT.hdr').asarray()
date_good2 = spectral.envi.open('dates1_2_RT.hdr').asarray()
date_good3 = spectral.envi.open('dates1_3_RT.hdr').asarray()
date_good4 = spectral.envi.open('dates1_4_RT.hdr').asarray()
date_good5 = spectral.envi.open('dates1_5_RT.hdr').asarray()


#%% remove_background (svc)

pick_point = []

filename ='dates_complex1_RT'  
dates = spectral.envi.open(filename+'.hdr').asarray()


select_bnands = [20,60,100]
false_color_bands = np.array(select_bnands)
img = dates[..., false_color_bands]
pick_ans = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
plt.Figure()
plt.imshow(img)
pick_temp = plt.ginput(25,show_clicks=True, timeout = 300)
plt.show()

for i in range(len(pick_temp)):
    pick_point.append(dates[int(pick_temp[i][1]),int(pick_temp[i][0]),:])
    
plt.close() 

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(pick_point, pick_ans)
dates_svc_sample = clf.predict(dates.reshape([-1,dates.shape[2]]))
plt.Figure()
plt.imshow(np.reshape(dates_svc_sample,([dates.shape[0],dates.shape[1]])))

#%% full band
dates_roi = peanut()


img_good_svc = []
img_good_roi = []
img_good_pca = []
img_bad_svc = []
img_bad_roi = []
img_bad_pca = []
img_key = []
img_value = []

for i in range(5):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_{i+1}_RT'  
    img_good = spectral.envi.open(filename + '.hdr').asarray()
      
    '''svc'''
    img_good_svc = clf.predict(img_good.reshape([-1,img_good.shape[2]]))
    img_good_svc = np.reshape(img_good_svc,([img_good.shape[0],img_good.shape[1]]))
    
    '''roi'''
    img_good_roi = dates_roi.test_peanut_roi(img_good, img_good_svc)
    
    for j in range(len(img_good_roi)):
        '''pca'''
        img_good_roi[j] = cv2.resize(img_good_roi[j], (300,300)) 
        
        img_key.append(img_good_roi[j])
        img_value.append(1)

for i in range(5):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{i+1}_RT'  
    img_bad = spectral.envi.open(filename + '.hdr').asarray()
    '''svc'''
    img_bad_svc = clf.predict(img_bad.reshape([-1,img_bad.shape[2]]))
    img_bad_svc = np.reshape(img_bad_svc,([img_bad.shape[0],img_bad.shape[1]]))
    '''roi'''
    img_bad_roi = dates_roi.test_peanut_roi(img_bad, img_bad_svc)
    
    for j in range(len(img_bad_roi)):
        '''pca'''
        img_bad_roi[j] = cv2.resize(img_bad_roi[j], (300,300))

        
        img_key.append(img_bad_roi[j])
        img_value.append(0)
        
        
img_value = np.array(img_value)


'''save npy'''
np.save('img_fullband.npy',img_key)
np.save('ans_fullband.npy', img_value)



#%%
dates_roi = peanut()
pca = PCA(n_components=80)

img_good_svc = []
img_good_roi = []
img_good_pca = []
img_bad_svc = []
img_bad_roi = []
img_bad_pca = []
img_key = []
img_value = []

for i in range(5):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_{i+1}_RT'  
    img_good = spectral.envi.open(filename + '.hdr').asarray()
      
    '''svc'''
    img_good_svc = clf.predict(img_good.reshape([-1,img_good.shape[2]]))
    img_good_svc = np.reshape(img_good_svc,([img_good.shape[0],img_good.shape[1]]))
    
    '''roi'''
    img_good_roi = dates_roi.test_peanut_roi(img_good, img_good_svc)
    
    for j in range(len(img_good_roi)):
        '''pca'''
        img_good_roi[j] = cv2.resize(img_good_roi[j], (300,300)) 
        img_good_pca = pca.fit_transform(np.reshape(img_good_roi[j],(img_good_roi[j].shape[0]*img_good_roi[j].shape[1],-1)))
        img_good_pca= np.reshape(img_good_pca,(300, 300, 80))
        
        img_key.append(img_good_pca)
        img_value.append(1)

for i in range(5):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{i+1}_RT'  
    img_bad = spectral.envi.open(filename + '.hdr').asarray()
    '''svc'''
    img_bad_svc = clf.predict(img_bad.reshape([-1,img_bad.shape[2]]))
    img_bad_svc = np.reshape(img_bad_svc,([img_bad.shape[0],img_bad.shape[1]]))
    '''roi'''
    img_bad_roi = dates_roi.test_peanut_roi(img_bad, img_bad_svc)
    
    for j in range(len(img_bad_roi)):
        '''pca'''
        img_bad_roi[j] = cv2.resize(img_bad_roi[j], (300,300))
        img_bad_pca = pca.fit_transform(np.reshape(img_bad_roi[j],(img_bad_roi[j].shape[0]*img_bad_roi[j].shape[1],-1)))
        img_bad_pca= np.reshape(img_bad_pca,(300,300,80))
        
        img_key.append(img_bad_pca)
        img_value.append(0)
        
        
img_value = np.array(img_value)


'''save npy'''
np.save('img.npy',img_key)
np.save('ans.npy', img_value)
#%% 2D CNN
img = np.load('img_fullband.npy',allow_pickle=True)
ans = np.load('ans_fullband.npy',allow_pickle=True)

ans = np_utils.to_categorical(ans)
origin_train, origin_test, ans_train, ans_test = train_test_split( img, ans, test_size=0.75, random_state=30)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(300, 300, 224), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2DTranspose(filters=64, kernel_size=3,activation='relu', padding='same'))
model.add(Flatten())  
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))


print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start = time.time()
model.fit(origin_train, ans_train, epochs=100, batch_size=50, validation_split = 0.05, verbose=1)

loss, accuracy = model.evaluate(origin_test, ans_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)

#save model
model.save("dates_try.h5")

#predict
    
dataset = "dates"
# load best weights
model.load_weights("dates_try.h5")
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
    print(end - start)
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
file_name = "dates_try.txt"

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


#%% 3D CNN

img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)

img = np.reshape(img, (img.shape[0],img.shape[1],img.shape[2],img.shape[3],1))

ans = np_utils.to_categorical(ans)
img_train, img_test, ans_train, ans_test = train_test_split( img, ans, test_size=0.3, random_state=50)

model = Sequential()
# model.add(Conv3D(filters=64, kernel_size=3, input_shape=(150, 150, 3, 1), activation='relu', padding='same'))
# model.add(Conv3D(filters=32, kernel_size=3,  activation='relu', padding='same'))
# model.add(MaxPool3D(pool_size=(2,2,1)))


# model.add(Conv3D(filters=32, kernel_size=3,  activation='relu', padding='same'))
# model.add(Conv3D(filters=64, kernel_size=3,  activation='relu', padding='same'))
# model.add(MaxPool3D(pool_size=(2,2,1)))


#model.add(Flatten())
#model.add(GlobalAveragePooling3D())
model.add(Conv3D(16, (3, 3, 3),input_shape=(150,150,3,1), padding = "same"))
model.add(LeakyReLU(alpha = 0.1))

model.add(Conv3D(32, (3, 3, 3), activation='relu', padding = "same"))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(GlobalAveragePooling3D())
model.add(Dense(2, activation='softmax'))



print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
MC = tf.keras.callbacks.ModelCheckpoint('model.h5', 
                                     monitor='val_loss', 
                                     verbose=0, 
                                     save_best_only=True, 
                                     mode='auto', 
                                     period=1)
ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=0, 
                                   patience=15, 
                                   verbose=1, 
                                   mode='auto', 
                                   baseline=None, 
                                   restore_best_weights=False)
RL = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.1, 
                                       patience=5, 
                                       verbose=1, 
                                       mode='auto', 
                                       min_delta=0.0001, 
                                       cooldown=0, 
                                       min_lr=10e-6)
history = model.fit(img_train, ans_train, epochs=100, batch_size=64,validation_split = 0.3, verbose=1, callbacks=[MC, ES, RL])

def show_train_history(history):
    plt.figure()
    plt.subplot(121)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Train History accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.legend(["train accuracy", "val accuracy"], loc=2)
    plt.subplot(122)    
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Train History loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.legend(["train loss", "val loss"], loc=2)
    plt.show()

show_train_history(history)


loss, accuracy = model.evaluate(img_test, ans_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)

#%% 1D+2D

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)
ans = np_utils.to_categorical(ans)
img_train, img_test, ans_train, ans_test = train_test_split( img, ans, test_size=0.4, random_state=45)


a = Input((150, 150, 3))
model_2D = Conv2D(16, kernel_size=(3, 3) , strides=(1,1), activation='relu',padding= 'same')(a)
model_2D = Conv2D(32, kernel_size=(3, 3) , strides=(1,1), activation='relu',padding= 'same')(model_2D)
model_2D = MaxPooling2D()(model_2D)





model_1D = tf.reshape(model_2D,(-1,model_2D.shape[1]*model_2D.shape[2],model_2D.shape[3]))

# 1
model_1D = Conv1D(16, kernel_size= 3 , strides=1, activation='relu')(model_1D)
model_1D = MaxPooling1D(pool_size= 2, strides=1)(model_1D)
# 2
model_1D = Conv1D(32, kernel_size= 3 , strides=1 , activation='relu')(model_1D)
model_1D = MaxPooling1D(pool_size= 2, strides=1)(model_1D)
# 3
model_1D = Conv1D(64, kernel_size= 3 , strides=1 , activation='relu')(model_1D)
model_1D = MaxPooling1D(pool_size= 2, strides=1)(model_1D)

model_1D = Flatten()(model_1D)
model_1D = Dense(2 , activation='softmax')(model_1D)

# model_2D = Flatten()(model_2D)
# model_2D = Dense(2 , activation='softmax')(model_2D)

model_final = Model(inputs=a, outputs=model_1D)
model_final.summary()
model_final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

MC = tf.keras.callbacks.ModelCheckpoint('model.h5', 
                                     monitor='val_loss', 
                                     verbose=0, 
                                     save_best_only=True, 
                                     mode='auto', 
                                     period=1)
ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                   min_delta=0, 
                                   patience=15, 
                                   verbose=1, 
                                   mode='auto', 
                                   baseline=None, 
                                   restore_best_weights=False)
RL = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.1, 
                                       patience=5, 
                                       verbose=1, 
                                       mode='auto', 
                                       min_delta=0.0001, 
                                       cooldown=0, 
                                       min_lr=10e-6)
history = model_final.fit(img_train, ans_train, epochs=100, batch_size=16,validation_split = 0.3, verbose=1, callbacks=[MC, ES, RL])
loss, accuracy = model_final.evaluate(img_test, ans_test)
def show_train_history(history):
    plt.figure()
    plt.subplot(121)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Train History accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.legend(["train accuracy", "val accuracy"], loc=2)
    plt.subplot(122)    
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Train History loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.legend(["train loss", "val loss"], loc=2)
    plt.show()

show_train_history(history)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)
