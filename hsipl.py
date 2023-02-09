#import package 

import os
import matplotlib.pyplot as plt
import spectral
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
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
#%% remove background
peanut_roi = peanut()
pca = PCA(n_components=3)

img_good_svc = []
img_good_roi = []
img_good_pca = []
img_bad_svc = []
img_bad_roi = []
img_bad_pca = []
img_key = []
img_value = []
pick_point = []


origin = spectral.envi.open('1_New-1.hdr').asarray()
origin = origin[0:200,:,:]

pick_ans = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
plt.Figure()
plt.imshow(origin[:,:,0])
pick_temp = plt.ginput(15,show_clicks=True,timeout = 300)
plt.show()

for i in range(len(pick_temp)):
    pick_point.append(origin[int(pick_temp[i][1]),int(pick_temp[i][0]),:])
    
plt.close()

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
result = clf.fit(pick_point, pick_ans)  

mask = clf.predict(np.reshape(origin,(200*409,25)))


#mask=clf.predict(origin.reshape(origin[0]*origin[1],25))
mask=mask.reshape(200,409)
plt.figure() 
plt.imshow(mask)


#%%



# for i in range(25):
#     '''load file'''
#     filename =f'C:/Users/user/Desktop/CNN/Desktop/CNN/peanut_cnn/g/g/{i+1}_New-1'  
#     img_good = spectral.envi.open(filename + '.hdr').asarray()
#     img_good = img_good[0:200,:,:]
#     '''svc'''
#     img_good_svc = clf.predict(img_good.reshape([-1,img_good.shape[2]]))
#     img_good_svc = np.reshape(img_good_svc,(200,409))
#     '''roi'''
#     img_good_roi = peanut_roi.test_peanut_roi(img_good, img_good_svc)
    
#     for j in range(len(img_good_roi)):
#         '''pca'''
#         img_good_roi[j] = cv2.resize(img_good_roi[j], (150,150)) 
#         img_good_pca = pca.fit_transform(np.reshape(img_good_roi[j],(img_good_roi[j].shape[0]*img_good_roi[j].shape[1],-1)))
#         img_good_pca= np.reshape(img_good_pca,(150,150,3))
        
#         img_key.append(img_good_pca)
#         img_value.append(1)

# for i in range(25):
#     '''load file'''
#     filename =f'C:/Users/user/Desktop/CNN/Desktop/CNN/peanut_cnn/b/b/{i+1}_New-1'  
#     img_bad = spectral.envi.open(filename + '.hdr').asarray()
#     img_bad = img_bad[0:200,:,:]
#     '''svc'''
#     img_bad_svc = clf.predict(img_bad.reshape([-1,img_bad.shape[2]]))
#     img_bad_svc = np.reshape(img_bad_svc,(200,409))
#     '''roi'''
#     img_bad_roi = peanut_roi.test_peanut_roi(img_bad, img_bad_svc)
    
#     for j in range(len(img_bad_roi)):
#         '''pca'''
#         img_bad_roi[j] = cv2.resize(img_bad_roi[j], (150,150))
#         img_bad_pca = pca.fit_transform(np.reshape(img_bad_roi[j],(img_bad_roi[j].shape[0]*img_bad_roi[j].shape[1],-1)))
#         img_bad_pca= np.reshape(img_bad_pca,(150,150,3))
        
#         img_key.append(img_bad_pca)
#         img_value.append(0)
        
        
# img_value = np.array(img_value)


# '''save npy'''
# np.save('img.npy',img_key)
# np.save('ans.npy', img_value)


 
   


'''ROI'''


'''data_roi = peanut()
peanut_roi = data_roi.test_peanut_roi(data, arr)
peanut_roi = np.array(peanut_roi)

print(peanut_roi)
plt.imshow(peanut_roi[0][:,:,0])'''

#%% 1D CNN
img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)

img = np.reshape(img, (img.shape[0],img.shape[1]*img.shape[2],img.shape[3]))

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
model.add(Conv1D(16, (3),input_shape=(150*150,3), padding = "same"))
model.add(LeakyReLU(alpha = 0.1))

model.add(Conv1D(32, (3), activation='relu', padding = "same"))

model.add(MaxPooling1D(pool_size=(2)))
model.add(GlobalAveragePooling1D())
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





#%% 2D CNN
img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)



ans = np_utils.to_categorical(ans)
img_train, img_test, ans_train, ans_test = train_test_split( img, ans, test_size=0.4, random_state=45)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, input_shape=(150, 150, 3), activation='relu', padding='same',strides =[2,2]))
model.add(Conv2D(filters=64, kernel_size=3,  activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(2, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(img_train, ans_train, epochs=100, batch_size=64, verbose=1)

loss, accuracy = model.evaluate(img_test, ans_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)

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


