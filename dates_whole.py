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
from sklearn.model_selection import KFold, GridSearchCV
from keras.utils.np_utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
import joblib
from sklearn.decomposition import IncrementalPCA
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# 載入高光譜數據
filename = 'dates4_4_RT'
dates = spectral.envi.open(filename+'.hdr').asarray()

#dates_enhanced = apply_adaptive_histogram_equalization(dates)

# 設定選取的波段和對應的類別
select_bnands = [20, 70, 100]
false_color_bands = np.array(select_bnands)
pick_ans = np.array([1]*30 + [0]*30)

# 選取樣本點
plt.Figure()
plt.imshow(dates[..., false_color_bands])
pick_temp = plt.ginput(60, show_clicks=True, timeout=300)
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
with open('remove_background_svc_model2.pkl', 'wb') as f:
    pickle.dump(clf, f)
#%% roi -> pca

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
#%% roi -> pca -> lda

def plot_pca_waveforms(pca_components, n_components):
    plt.figure(figsize=(12, 6))
    for i in range(n_components):
        plt.plot(pca_components[i], label=f'PC{i+1}')
    plt.xlabel('Spectral Bands')
    plt.ylabel('PCA Component')
    plt.title('PCA Waveforms')
    plt.legend()
    plt.show()

def process_images(file_path, count, img_label, clf, img_key, img_value):
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
            '''resize'''
            img_roi[j] = cv2.resize(img_roi[j], (200, 200))
            print("img_roi shape:", img_roi[j].shape)
            img_key.append(img_roi[j])
            img_value.append(img_label)
            
start_time = time.time()
with open('remove_background_svc_model2.pkl', 'rb') as f:
    clf = pickle.load(f)

dates_roi = peanut()
img_key = []
img_value = []

process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_{}_RT', 25, 0, clf, img_key, img_value) #正常
process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{}_RT', 18, 1, clf, img_key, img_value) #裂紋
process_images('C:/Users/user/Desktop/YuJie/dates_data/0306蜜棗/dates3_{}_RT', 20, 2, clf, img_key, img_value) #腐敗
process_images('C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_{}_RT', 15, 3, clf, img_key, img_value) #炭砠

img_key = np.array(img_key)
img_value = np.array(img_value)

'''PCA'''
pca_n_components = 60
pca = PCA(n_components=pca_n_components)
img_pca = pca.fit_transform(img_key.reshape(img_key.shape[0], -1))
print(f'Selected PCA bands: {list(range(1, pca_n_components+1))}')
plot_pca_waveforms(pca.components_, pca_n_components)

'''LDA'''
lda_n_components = 3
lda = LDA(n_components=lda_n_components)
img_lda = lda.fit_transform(img_pca, img_value)

'''save npy'''
np.save('img_4class_pca60_lda3.npy', img_lda)
np.save('ans_4class_pca60_lda3.npy', img_value)

end_time = time.time()
print('前處理時間：', end_time - start_time, '秒')



#%% roi -> lda

def process_images(file_path, count, img_label, clf, img_key, img_value):
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
            '''resize'''
            img_roi[j] = cv2.resize(img_roi[j], (200, 200))
            print("img_roi shape:", img_roi[j].shape)
            img_key.append(img_roi[j])
            img_value.append(img_label)

start_time = time.time()
with open('remove_background_svc_model2.pkl', 'rb') as f:
    clf = pickle.load(f)

dates_roi = peanut()
img_key = []
img_value = []

process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_{}_RT', 25, 0, clf, img_key, img_value) #正常
process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{}_RT', 18, 1, clf, img_key, img_value) #裂紋
process_images('C:/Users/user/Desktop/YuJie/dates_data/0306蜜棗/dates3_{}_RT', 20, 2, clf, img_key, img_value) #腐敗
process_images('C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_{}_RT', 15, 3, clf, img_key, img_value) #炭砠

img_key = np.array(img_key)
img_value = np.array(img_value)

'''LDA'''
lda_n_components = 3
# 初始化 IncrementalPCA
ipca = IncrementalPCA(n_components=lda_n_components)

# 將資料集分成5個part
batches = np.array_split(img_key.reshape(img_key.shape[0], -1), 5)

# 分批次訓練
for batch in batches:
    ipca.partial_fit(batch)

# 使用訓練好的 IncrementalPCA 對 img_key 進行降维
img_ipca = ipca.transform(img_key.reshape(img_key.shape[0], -1))

# 使用 LDA 進行降维
lda = LDA(n_components=lda_n_components)
img_lda = lda.fit_transform(img_ipca, img_value)

np.save('img_4class_lda.npy', img_lda)
np.save('ans_4class_lda.npy', img_value)

# lda = LDA(n_components=lda_n_components)
# img_lda = lda.fit_transform(img_key.reshape(img_key.shape[0], -1), img_value)

# '''save npy'''
# np.save('img_4class_lda.npy', img_lda)
# np.save('ans_4class_lda.npy', img_value)

end_time = time.time()
print('前處理時間：', end_time - start_time, '秒')


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
img = np.load('img_4class_pca30_lda.npy',allow_pickle=True)
ans = np.load('ans_4class_pca30_lda.npy',allow_pickle=True)

# 定義 KFold，將數據集分成 5 折
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# 建立增強後的數據
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

# 定義模型
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 30), activation='relu', padding='same'))
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

model.save("dates_4class_model3_pca30_lda_1.h5")

#predict
    
dataset = "dates"
# load best weights
model.load_weights("dates_4class_model3_pca30_lda_1.h5")
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
file_name = "dates_4class_model3_pca30_lda_1.txt"

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
#%% random forest

# 加載數據
img = np.load('img_4class_pca90_lda1.npy', allow_pickle=True)
ans = np.load('ans_4class_pca90_lda1.npy', allow_pickle=True)

best_mean_accuracy = 0
best_random_state = 0
best_rf_model = None

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

use_grid_search = False  #  True 以使用網格搜索， False 則使用普通隨機森林

for random_state in range(30):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    total_accuracy = 0
    
    for train_idx, test_idx in kf.split(img, ans):
        origin_train, origin_test = img[train_idx], img[test_idx]
        ans_train, ans_test = ans[train_idx], ans[test_idx]

        if use_grid_search:
            rf_clf = RandomForestClassifier(random_state=0)
            grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            start_time = time.time()
            grid_search.fit(origin_train, ans_train)
            end_time = time.time()
            best_rf_clf = grid_search.best_estimator_
        else:
            best_rf_clf = RandomForestClassifier(random_state=0)
            start_time = time.time()
            best_rf_clf.fit(origin_train, ans_train)
            end_time = time.time()

        ans_pred = best_rf_clf.predict(origin_test)

        accuracy = accuracy_score(ans_test, ans_pred)
        print('Test:')
        print('Accuracy:', accuracy)
        total_accuracy += accuracy

    mean_accuracy = total_accuracy / 5
    print('Mean accuracy for random state', random_state, ':', mean_accuracy)

    if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        best_random_state = random_state
        best_rf_model = best_rf_clf

    print('Prediction time:', end_time - start_time, 'seconds')

print('Best random state:', best_random_state)
print('Best mean accuracy:', best_mean_accuracy)

joblib.dump(best_rf_model, 'best_pca90_lda1_rf_model.pkl')
    
#%% k-nn

img = np.load('img_4class_pca90_lda1.npy', allow_pickle=True)
ans = np.load('ans_4class_pca90_lda1.npy', allow_pickle=True)

param_grid = {'n_neighbors': list(range(1, 31))}

best_mean_accuracy = 0
best_random_state = 0
best_knn_model = None

for random_state in range(10):  # 進行10次隨機迭代
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    total_accuracy = 0
    
    for train_idx, test_idx in kf.split(img, ans):
        origin_train, origin_test = img[train_idx], img[test_idx]
        ans_train, ans_test = ans[train_idx], ans[test_idx]

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=6, scoring='accuracy')

        start_time = time.time()
        grid_search.fit(origin_train, ans_train)
        end_time = time.time()

        print("Best parameters found:", grid_search.best_params_)

        ans_pred = grid_search.predict(origin_test)

        accuracy = accuracy_score(ans_test, ans_pred)
        print('Test:')
        print('Accuracy:', accuracy)
        print('Test time:', end_time - start_time, 'seconds')

        total_accuracy += accuracy

    mean_accuracy = total_accuracy / 5  # 計算平均交叉驗證準確率
    print('Mean accuracy for random state', random_state, ':', mean_accuracy)

    if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        best_random_state = random_state
        best_knn_model = grid_search.best_estimator_

print('Best random state:', best_random_state)
print('Best mean accuracy:', best_mean_accuracy)

joblib.dump(best_knn_model, 'best_pca90_lda1_knn_model.pkl')

#%% 集成學習(random_forest、K-nn)
img = np.load('img_4class_pca60_lda.npy', allow_pickle=True)
ans = np.load('ans_4class_pca60_lda.npy', allow_pickle=True)

best_mean_accuracy = 0
best_random_state = 0
best_voting_model = None

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

use_grid_search = False  #  True 以使用網格搜索， False 則使用普通隨機森林

for random_state in range(30):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    total_accuracy = 0
    
    for train_idx, test_idx in kf.split(img, ans):
        origin_train, origin_test = img[train_idx], img[test_idx]
        ans_train, ans_test = ans[train_idx], ans[test_idx]

        if use_grid_search:
            rf_clf = RandomForestClassifier(random_state=0)
            grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            start_time = time.time()
            grid_search.fit(origin_train, ans_train)
            end_time = time.time()
            best_rf_clf = grid_search.best_estimator_
        else:
            best_rf_clf = RandomForestClassifier(random_state=0)
            best_knn_clf = KNeighborsClassifier(n_neighbors=5)
            best_voting_clf = VotingClassifier(
                estimators=[
                    ('rf', best_rf_clf), 
                    ('knn', best_knn_clf)
                ],
                voting='soft'
            )
            start_time = time.time()
            best_voting_clf.fit(origin_train, ans_train)
            end_time = time.time()

        ans_pred = best_voting_clf.predict(origin_test)

        accuracy = accuracy_score(ans_test, ans_pred)
        print('Test:')
        print('Accuracy:', accuracy)
        total_accuracy += accuracy

    mean_accuracy = total_accuracy / 5
    print('Mean accuracy for random state', random_state, ':', mean_accuracy)

    if mean_accuracy > best_mean_accuracy:
        best_mean_accuracy = mean_accuracy
        best_random_state = random_state
        best_voting_model = best_voting_clf

    print('Prediction time:', end_time - start_time, 'seconds')

print('Best random state:', best_random_state)
print('Best mean accuracy:', best_mean_accuracy)

joblib.dump(best_voting_model, 'best_pca60_lda_voting_model1.pkl')


