import os
import matplotlib.pyplot as plt
import spectral
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow.keras.layers import Conv3D,Conv2D,MaxPool2D, MaxPooling3D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D, GlobalAveragePooling3D,Conv1D, MaxPool1D, GlobalAveragePooling1D, Conv2DTranspose, Conv3DTranspose,BatchNormalization,Input, Concatenate 
import cv2
import pickle
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
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
#%% 
# filename_good =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_1_RT'  
# img_good = spectral.envi.open(filename_good + '.hdr').asarray()
filename_crack =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_1_RT'  
img_crack = spectral.envi.open(filename_crack + '.hdr').asarray()
filename_anthrax =f'C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_2_RT'  
img_anthrax = spectral.envi.open(filename_anthrax + '.hdr').asarray()

result = np.concatenate((img_crack, img_anthrax),axis=0)
#%% check file size

date_good1 = spectral.envi.open('dates3_1_RT.hdr').asarray()
date_good2 = spectral.envi.open('dates3_2_RT.hdr').asarray()
date_good3 = spectral.envi.open('dates3_3_RT.hdr').asarray()
date_good4 = spectral.envi.open('dates3_4_RT.hdr').asarray()
date_good5 = spectral.envi.open('dates3_5_RT.hdr').asarray()
select_bnands = [20,65,100]
false_color_bands = np.array(select_bnands)
img = date_good5[..., false_color_bands]
plt.Figure()
plt.imshow(img)

#%% remove_background (svc)

pick_point = []

filename ='dates4_4_RT'  
dates = spectral.envi.open(filename+'.hdr').asarray()
select_bnands = [20,65,100]

false_color_bands = np.array(select_bnands)
img = dates[..., false_color_bands]
pick_ans = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #20個非背景 25個背景
plt.Figure()
plt.imshow(img)
pick_temp = plt.ginput(45,show_clicks=True, timeout = 300)
plt.show()

for i in range(len(pick_temp)):
    pick_point.append(dates[int(pick_temp[i][1]),int(pick_temp[i][0]),:])
    
plt.close() 

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(pick_point, pick_ans)
dates_svc_sample = clf.predict(dates.reshape([-1,dates.shape[2]]))
plt.Figure()
plt.imshow(np.reshape(dates_svc_sample,([dates.shape[0],dates.shape[1]])))

#%%
# 載入高光譜數據
filename = 'dates4_4_RT'
dates = spectral.envi.open(filename+'.hdr').asarray()

# 設定選取的波段和對應的類別
select_bnands = [20, 65, 100]
false_color_bands = np.array(select_bnands)
pick_ans = np.array([1]*25 + [0]*25)

# 選取樣本點
plt.Figure()
plt.imshow(dates[..., false_color_bands])
pick_temp = plt.ginput(50, show_clicks=True, timeout=300)
plt.show()

# 儲存樣本點
pick_point = []
for i in range(len(pick_temp)):
    pick_point.append(dates[int(pick_temp[i][1]), int(pick_temp[i][0]), :])
    
plt.close()

# 使用SVC去除背景
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(pick_point, pick_ans)
dates_svc_sample = clf.predict(dates.reshape([-1, dates.shape[2]]))

# 顯示去除背景後的影像
plt.Figure()
plt.imshow(np.reshape(dates_svc_sample, [dates.shape[0], dates.shape[1]]))

# 儲存模型
with open('remove_background_svc_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
#%%

def process_images(file_path, count, img_label, clf, pca, n_components, img_key, img_value):
    for i in range(count):
        '''load file'''
        filename = file_path.format(i + 1)
        img = spectral.envi.open(filename + '.hdr').asarray()

        '''svc'''
        img_svc = clf.predict(img.reshape([-1, img.shape[2]]))
        img_svc = np.reshape(img_svc, ([img.shape[0], img.shape[1]]))

        '''roi'''
        img_roi = dates_roi.test_peanut_roi(img, img_svc)

        for j in range(len(img_roi)):
            '''pca'''
            img_roi[j] = cv2.resize(img_roi[j], (200, 200))
            img_pca = pca.fit_transform(np.reshape(img_roi[j], (img_roi[j].shape[0] * img_roi[j].shape[1], -1)))
            img_pca = np.reshape(img_pca, (200, 200, n_components))

            img_key.append(img_pca)
            img_value.append(img_label)

with open('remove_background_svc_model.pkl', 'rb') as f:
    clf = pickle.load(f)

n_components = 10
dates_roi = peanut()
pca = PCA(n_components)
img_key = []
img_value = []

process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_{}_RT', 25, 0, clf, pca, n_components, img_key, img_value) #正常
process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{}_RT', 18, 1, clf, pca, n_components, img_key, img_value) #裂紋
process_images('C:/Users/user/Desktop/YuJie/dates_data/0306蜜棗/dates3_{}_RT', 20, 2, clf, pca, n_components, img_key, img_value) #腐敗
process_images('C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_{}_RT', 15, 3, clf, pca, n_components, img_key, img_value) #炭砠

'''save npy'''
np.save('img_4class_pca10.npy',img_key)
np.save('ans_4class_pca10.npy', img_value)
#%% roi -> pca

# 讀取模型
with open('remove_background_svc_model.pkl', 'rb') as f:
    clf = pickle.load(f)
    
dates_roi = peanut()
pca = PCA(n_components=30)

img_good_svc = []
img_good_roi = []
img_good_pca = []
img_crack_svc = []
img_crack_roi = []
img_crack_pca = []
img_anthrax_svc = []
img_anthrax_roi = []
img_anthrax_pca = []
img_overripe_svc = []
img_overripe_roi = []
img_overripe_pca = []
img_key = []
img_value = []

for i in range(25):
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
        img_good_roi[j] = cv2.resize(img_good_roi[j], (200,200)) 
        img_good_pca = pca.fit_transform(np.reshape(img_good_roi[j],(img_good_roi[j].shape[0]*img_good_roi[j].shape[1],-1)))
        img_good_pca= np.reshape(img_good_pca,(200, 200, 30))
        
        img_key.append(img_good_pca)
        img_value.append(0)

for i in range(18):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{i+1}_RT'  
    img_crack = spectral.envi.open(filename + '.hdr').asarray()
    '''svc'''
    img_crack_svc = clf.predict(img_crack.reshape([-1,img_crack.shape[2]]))
    img_crack_svc = np.reshape(img_crack_svc,([img_crack.shape[0],img_crack.shape[1]]))
    '''roi'''
    img_crack_roi = dates_roi.test_peanut_roi(img_crack, img_crack_svc)
    
    for j in range(len(img_crack_roi)):
        '''pca'''
        img_crack_roi[j] = cv2.resize(img_crack_roi[j], (200,200))
        img_crack_pca = pca.fit_transform(np.reshape(img_crack_roi[j],(img_crack_roi[j].shape[0]*img_crack_roi[j].shape[1],-1)))
        img_crack_pca= np.reshape(img_crack_pca,(200,200,30))
        
        img_key.append(img_crack_pca)
        img_value.append(1)


for i in range(15):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_{i+1}_RT'  
    img_anthrax = spectral.envi.open(filename + '.hdr').asarray()
    '''svc'''
    img_anthrax_svc = clf.predict(img_anthrax.reshape([-1,img_anthrax.shape[2]]))
    img_anthrax_svc = np.reshape(img_anthrax_svc,([img_anthrax.shape[0],img_anthrax.shape[1]]))
    '''roi'''
    img_anthrax_roi = dates_roi.test_peanut_roi(img_anthrax, img_anthrax_svc)
    
    for j in range(len(img_anthrax_roi)):
        '''pca'''
        img_anthrax_roi[j] = cv2.resize(img_anthrax_roi[j], (200,200))
        img_anthrax_pca = pca.fit_transform(np.reshape(img_anthrax_roi[j],(img_anthrax_roi[j].shape[0]*img_anthrax_roi[j].shape[1],-1)))
        img_anthrax_pca= np.reshape(img_anthrax_pca,(200,200,30))
        
        img_key.append(img_anthrax_pca)
        img_value.append(2)
        
for i in range(20):
    '''load file'''
    filename =f'C:/Users/user/Desktop/YuJie/dates_data/0306蜜棗/dates3_{i+1}_RT'  
    img_overripe = spectral.envi.open(filename + '.hdr').asarray()
    '''svc'''
    img_overripe_svc = clf.predict(img_overripe.reshape([-1,img_overripe.shape[2]]))
    img_overripe_svc = np.reshape(img_overripe_svc,([img_overripe.shape[0],img_overripe.shape[1]]))
    '''roi'''
    img_overripe_roi = dates_roi.test_peanut_roi(img_overripe, img_overripe_svc)
    
    for j in range(len(img_anthrax_roi)):
        '''pca'''
        img_overripe_roi[j] = cv2.resize(img_overripe_roi[j], (200,200))
        img_overripe_pca = pca.fit_transform(np.reshape(img_overripe_roi[j],(img_overripe_roi[j].shape[0]*img_overripe_roi[j].shape[1],-1)))
        img_overripe_pca= np.reshape(img_overripe_pca,(200,200,30))
        
        img_key.append(img_overripe_pca)
        img_value.append(3)
        
img_value = np.array(img_value)


'''save npy'''
np.save('img_4class_pca30.npy',img_key)
np.save('ans_4class_pca30.npy', img_value)
#%% model try1 

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 30), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())

#%% model try2

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 30), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())

#%% model try3

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 30), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())

#%% 2D CNN
img = np.load('img_4class_pca10.npy',allow_pickle=True)
ans = np.load('ans_4class_pca10.npy',allow_pickle=True)

# 定義 KFold 分割器，將數據集分成 5 折
kf = KFold(n_splits=6, shuffle=True, random_state=0)

# 建立增強後的數據產生器
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

# 定義模型
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 10), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 使用 KFold 進行交叉驗證
for train_idx, test_idx in kf.split(img, ans):
    # 將數據集按照 KFold 分割的索引分割為訓練集和測試集
    origin_train, origin_test = img[train_idx], img[test_idx]
    ans_train, ans_test = ans[train_idx], ans[test_idx]
    
    # 將整數標籤轉換為 one-hot 編碼的標籤
    ans_train = to_categorical(ans_train, num_classes=4)
    ans_test = to_categorical(ans_test, num_classes=4)
    
    # 使用增強後的數據產生器進行模型訓練
    datagen.fit(origin_train)
    model.fit(datagen.flow(origin_train, ans_train, batch_size=100), epochs=100, validation_data=(origin_test, ans_test))
    
    # 評估模型性能
    loss, accuracy = model.evaluate(origin_test, ans_test)
    print('Test:')
    print('Loss:', loss)
    print('Accuracy:', accuracy)

model.save("dates_4class_model3_pca10.h5")

#predict
    
dataset = "dates"
# load best weights
model.load_weights("dates_4class_model3_pca10.h5")
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
        target_names = ['good','crack', 'decay' , 'anthrax']
    
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
file_name = "dates_4class_model3_pca10.txt"

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


#%%
# ans = np_utils.to_categorical(ans)
# origin_train, origin_test, ans_train, ans_test = train_test_split( img, ans, test_size=0.2, random_state=0)


# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 50), activation='relu', padding = 'same'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding = 'same'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding = 'same'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Conv2DTranspose(filters=64, kernel_size=3,activation='relu', padding='same'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(3, activation='softmax'))


# print(model.summary())
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# start = time.time()
# model.fit(origin_train, ans_train, epochs=1000, batch_size=100,validation_split = 0.05,verbose=1)

# loss, accuracy = model.evaluate(origin_test, ans_test)
# print('Test:')
# print('Loss:', loss)
# print('Accuracy:', accuracy)

#save model
#%% 3D CNN

img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)



img = np.reshape(img, (img.shape[0],img.shape[1],img.shape[2],img.shape[3],1))

ans = np_utils.to_categorical(ans)
origin_train, origin_test, ans_train, ans_test = train_test_split( img, ans, test_size=0.4, random_state=0) 

model = Sequential()
model.add(Conv3D(filters=16, kernel_size=3, input_shape=(200, 200, 50, 1), activation='relu'))
model.add(MaxPool3D(pool_size=2))
#model.add(Conv3DTranspose(filters=32, kernel_size=3,activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))



print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(origin_train, ans_train, epochs=1000, batch_size=100,validation_split = 0.05, verbose=1)

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


loss, accuracy = model.evaluate(origin_test, ans_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)

#%% 2D+3D (並聯)
img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)

origin_train_2D, origin_test_2D, ans_train_2D, ans_test_2D = train_test_split( img, ans, test_size=0.6, random_state=0)
origin_train_3D, origin_test_3D, ans_train_3D, ans_test_3D = train_test_split( img, ans, test_size=0.6, random_state=0)

ans_train = np_utils.to_categorical(ans_train_2D)
ans_test = np_utils.to_categorical(ans_test_2D)

in_2D = Input((150, 150, 30))
model_2D = Conv2D(16, kernel_size=(3,3) , strides=(1,1), activation='relu', padding='valid')(in_2D)
#model_2D = MaxPooling2D(pool_size=(3,3) , strides=(1,1))(model_2D)
model_2D = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = MaxPooling2D(pool_size=(2,2) , strides=(1,1), padding='same')(model_2D)
model_2D = Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = Conv2DTranspose(64, kernel_size=(5,5) , strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = GlobalAveragePooling2D()(model_2D)


#------------------------------3D-----------------------------------------
in_3D = Input((150, 150, 30, 1))
model_3D = Conv3D(8, kernel_size=(3,3,7), strides=(1,1,1), padding='same',activation='relu')(in_3D)
model_3D = Conv3D(16, kernel_size=(3,3,5), strides=(1,1,1), padding='same',activation='relu')(model_3D)
model_3D = MaxPooling3D(pool_size=(3,3,3) , strides=(1,1,1), padding='same')(model_3D)
model_3D = Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1),  padding='same', activation='relu')(model_3D)
model_3D = GlobalAveragePooling3D()(model_3D)

merged = Concatenate()([model_2D, model_3D])
output = Dense(64, activation='relu')(merged)
output = Dense(16, activation='relu')(output)
output = Dense(3, activation='softmax')(output)

model_combined = Model(inputs=[in_2D, in_3D], outputs=[output])
adam = Adam(lr=0.001, decay=1e-06)
start = time.time()
model_combined.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model_combined.summary()
history = model_combined.fit([origin_train_2D, origin_train_3D], ans_train, validation_split=0.05, epochs=1000, batch_size=30)
print("Total training time: ", time.time() - start, "seconds")
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
        target_names = ['good','crack', 'anthrax']
    
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
#%% 2D+3D (串聯)
img = np.load('img.npy',allow_pickle=True)
ans = np.load('ans.npy',allow_pickle=True)

origin_train_2D, origin_test_2D, ans_train_2D, ans_test_2D = train_test_split( img, ans, test_size=0.75, random_state=0)

ans_train = np_utils.to_categorical(ans_train_2D)
ans_test = np_utils.to_categorical(ans_test_2D)

in_2D = Input((150, 150, 10))
model_2D = Conv2D(16, kernel_size=(3,3) , strides=(1,1), activation='relu', padding='valid')(in_2D)
#model_2D = MaxPooling2D(pool_size=(3,3) , strides=(1,1))(model_2D)
model_2D = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = MaxPooling2D(pool_size=(2,2) , strides=(1,1), padding='same')(model_2D)
model_2D = Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = Conv2DTranspose(64, kernel_size=(5,5) , strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = model_2D[...,None]

#------------------------------3D-----------------------------------------
model_3D = Conv3D(8, kernel_size=(3,3,7), strides=(1,1,1), padding='same',activation='relu')(model_2D)
model_3D = Conv3D(16, kernel_size=(3,3,5), strides=(1,1,1), padding='same',activation='relu')(model_3D)
model_3D = MaxPooling3D(pool_size=(3,3,3) , strides=(1,1,1), padding='same')(model_3D)
model_3D = Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1),  padding='same', activation='relu')(model_3D)
model_3D = GlobalAveragePooling3D()(model_3D)

output = Dense(64, activation='relu')(model_3D)
output = Dense(16, activation='relu')(output)
output = Dense(3, activation='softmax')(output)

model_combined = Model(inputs=in_2D, outputs=[output])
adam = Adam(lr=0.001, decay=1e-06)
model_combined.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model_combined.summary()
history = model_combined.fit(origin_train_2D, ans_train, validation_split=0.05, epochs=300, batch_size=100)
print("Total training time: ", time.time() - start, "seconds")
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

