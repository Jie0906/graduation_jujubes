#%% load package
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
from peanut_class import peanut #SVC的方法
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
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score, recall_score, precision_score
#%% 確認檔案大小用
date_good1 = spectral.envi.open('dates1_2_RT.hdr').asarray()
select_bnands = [30,60,90]
false_color_bands = np.array(select_bnands)
img = date_good1[..., false_color_bands]
plt.Figure()
plt.imshow(img)

#%% 繪出各類1顆的光譜反射率圖

# 選取每類一顆蜜棗的數據並獲取中心點的光譜資料
dates_class1 = spectral.envi.open('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_1_RT.hdr').asarray()[int(1306/2), int(1024/2), :] #正常
dates_class2 = spectral.envi.open('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_1_RT.hdr').asarray()[int(1306/2), int(1024/2), :] #裂紋
dates_class3 = spectral.envi.open('C:/Users/user/Desktop/YuJie/dates_data/0306蜜棗/dates3_1_RT.hdr').asarray()[int(1306/2), int(1024/2), :] #腐敗
dates_class4 = spectral.envi.open('C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_1_RT.hdr').asarray()[int(1306/2), int(1024/2), :] #炭疽

wavelength_start = 400  # 波長起始值
wavelength_end = 1000  # 波長結束值
num_bands = 224  # 波段數量

# 建立從波段index到波長的映射
wavelengths = np.linspace(wavelength_start, wavelength_end, num_bands)

# X為波長、Y為反射率
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, dates_class1, label='Normal',color = 'green')
plt.plot(wavelengths, dates_class2, label='Crack',color = 'red')
plt.plot(wavelengths, dates_class3, label='Decay',color = 'orange')
plt.plot(wavelengths, dates_class4, label='Anthrax ',color = 'purple')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Reflectance Spectra for One jujube from Each Class')
plt.legend()
plt.show()

#X為波段數、Y為反射率
# plt.figure(figsize=(12, 6))
# plt.plot(wavelengths, dates_class1, label='Normal',color = 'green')
# plt.plot(wavelengths, dates_class2, label='Crack',color = 'red')
# plt.plot(wavelengths, dates_class3, label='Decay',color = 'orange')
# plt.plot(wavelengths, dates_class4, label='Anthrax ',color = 'purple')
# plt.xlabel('Spectral Bands')
# plt.ylabel('Reflectance')
# plt.title('Reflectance Spectra for One jujube from Each Class')
# plt.legend()
# plt.show()


#%% 前處理 去背景 SVC(SVM下的方法)

# 載入高光譜數據
filename = 'dates4_4_RT' #選背景點/非背景點用的圖 隨便一張都可
dates = spectral.envi.open(filename+'.hdr').asarray()

# 設定選取的波段和對應的類別
select_bnands = [20, 70, 100] #偽色圖
false_color_bands = np.array(select_bnands)
pick_ans = np.array([1]*30 + [0]*30)

# 選取樣本點
plt.Figure()
plt.imshow(dates[..., false_color_bands])
pick_temp = plt.ginput(60, show_clicks=True, timeout=300) #樣本60個點 選取時間300秒
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

#%% 計算各%數所需主成分

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

# 將所有維度都作為PCA的n_components參數，進行PCA轉換
pca = PCA(n_components=None)
pca.fit(img_key.reshape(img_key.shape[0], -1))

# 計算累積解釋的變異比例
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 找到累積解釋變異比例達到80%、85%、90%、95%的主成分數量
for target_variance in [0.8, 0.85, 0.9, 0.95]:
    n_components = np.argmax(cumulative_explained_variance >= target_variance) + 1
    print(f"To achieve {target_variance * 100}% of total variance, {n_components} principal components are needed.")

# 繪製累積解釋變異比例隨著主成分數量增加的曲線
plt.figure(figsize=(10, 5))
plt.plot(cumulative_explained_variance)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()
    
#%% 前處理 roi -> pca -> lda
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
pca_n_components =30
pca = PCA(n_components=pca_n_components)
img_pca = pca.fit_transform(img_key.reshape(img_key.shape[0], -1))

# 計算每個主成分解釋的變異比例
explained_variance_ratio = pca.explained_variance_ratio_

# 計算累積解釋的變異比例
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 繪製每個主成分解釋的變異比例的長條圖
plt.figure(figsize=(10, 5))
plt.bar(range(pca_n_components), explained_variance_ratio, alpha=0.5, label='Individual explained variance')
plt.step(range(pca_n_components), cumulative_explained_variance, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 印出總體解釋的變異比例
print("Total explained variance ratio by", pca_n_components, "principal components:", cumulative_explained_variance[-1])

'''LDA'''
lda_n_components = 3
lda = LDA(n_components=lda_n_components)
img_lda = lda.fit_transform(img_pca, img_value)

'''save npy'''
np.save('img_4class_pca30_lda2.npy', img_lda)
np.save('ans_4class_pca30_lda2.npy', img_value)

end_time = time.time()
print('前處理時間：', end_time - start_time, '秒')

#%% 前處理 roi -> pca

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

n_components = 90 #要降到的波段數
dates_roi = peanut()
pca = PCA(n_components)
img_key = []
img_value = []

process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/一級/dates1_{}_RT', 25, 0, clf, pca, n_components, img_key, img_value) #正常
process_images('C:/Users/user/Desktop/YuJie/dates_data/0204蜜棗/二級/dates2_{}_RT', 18, 1, clf, pca, n_components, img_key, img_value) #裂紋
process_images('C:/Users/user/Desktop/YuJie/dates_data/0306蜜棗/dates3_{}_RT', 20, 2, clf, pca, n_components, img_key, img_value) #腐敗
process_images('C:/Users/user/Desktop/YuJie/dates_data/0216蜜棗/四級/dates4_{}_RT', 15, 3, clf, pca, n_components, img_key, img_value) #炭砠

'''save npy'''
np.save('img_4class_pca90.npy',img_key)
np.save('ans_4class_pca90.npy', img_value)



#%% 前處理 roi -> lda

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
end_time = time.time()
print('前處理時間：', end_time - start_time, '秒')


#%% 2D CNN model try1 

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

#%% 2D CNN model try2

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

#%% 2D CNN model try3

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 30), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())

#%% 後處理 2D CNN
img = np.load('img_4class_pca60.npy',allow_pickle=True)
ans = np.load('ans_4class_pca60.npy',allow_pickle=True)

# 交叉驗證，定義 KFold，將數據集分成 5 折
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
model.add(Conv2D(filters=16, kernel_size=3, input_shape=(200, 200, 60), activation='relu', padding='same'))
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

model.save("dates_4class_pca60.h5")

#predict
    
dataset = "dates"
# load best weights
model.load_weights("dates_4class_pca60.h5")  #這東西名稱會跟461一樣
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
    if name == 'dates':
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
file_name = "dates_4class_pca60.txt"

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
#%% 後處理 集成學習(結合隨機森林、K-NN)

# 網格搜索各參數
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


def fit_model(train_data, train_labels, use_grid_search):
    if use_grid_search: #若網格搜索為True
        rf_clf = RandomForestClassifier(random_state=0)
        grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        start_time = time.time()
        grid_search.fit(train_data, train_labels)
        return grid_search.best_estimator_, time.time() - start_time
    else: #使用預設參數
        rf_clf = RandomForestClassifier(random_state=0)
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf_clf), 
                ('knn', knn_clf)
            ],
            voting='soft'
        )
        start_time = time.time()
        voting_clf.fit(train_data, train_labels)
        return voting_clf, time.time() - start_time

def evaluate_model(model, test_data, test_labels): #計算各項指標
    pred = model.predict(test_data)
    accuracy = accuracy_score(test_labels, pred)
    kappa = cohen_kappa_score(test_labels, pred)
    iou = jaccard_score(test_labels, pred, average='weighted')
    return accuracy, kappa, iou

def perform_cross_validation(data, labels, random_state, use_grid_search):
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state) #做5次交叉驗證
    total_accuracy = 0
    total_kappa = 0
    iou_scores = []
    for train_idx, test_idx in kf.split(data, labels):
        train_data, train_labels = data[train_idx], labels[train_idx]
        test_data, test_labels = data[test_idx], labels[test_idx]
        model, training_time = fit_model(train_data, train_labels, use_grid_search)
        accuracy, kappa, iou = evaluate_model(model, test_data, test_labels)
        total_accuracy += accuracy
        total_kappa += kappa
        iou_scores.append(iou)
    return total_accuracy / 5, total_kappa / 5, sum(iou_scores) / len(iou_scores), model

def main():
    #載入前處理後的.npy檔
    img = np.load('img_4class_pca90_lda1.npy', allow_pickle=True)
    ans = np.load('ans_4class_pca90_lda1.npy', allow_pickle=True)

    best_mean_accuracy = 0
    best_kappa = 0
    best_iou = 0
    best_model = None


    for random_state in range(10): #隨機使用10個亂數種子
        mean_accuracy, mean_kappa, mean_iou, model = perform_cross_validation(img, ans, random_state, use_grid_search=False) #是否啟用網格搜索

        if mean_iou > best_iou:
            best_iou = mean_iou
        if mean_kappa > best_kappa:
            best_kappa = mean_kappa
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            best_model = model


    best_pred = best_model.predict(img)
    best_f1 = f1_score(ans, best_pred, average='weighted')
    best_recall = recall_score(ans, best_pred, average='weighted')
    best_precision = precision_score(ans, best_pred, average='weighted')
    
    #印出各項指標
    print('--- ---')
    print('Best Mean IOU:', best_iou)
    print('Best F1 score:', best_f1)
    print('Best Recall:', best_recall)
    print('Best Precision:', best_precision)
    print('Best Kappa:', best_kappa)
    print('Best mean accuracy:', best_mean_accuracy)

    np.savetxt('best_pred.txt', best_pred, fmt='%d')
    joblib.dump(best_model, 'best_pca90_lda_voting_model1.pkl') #要存成的模型名稱

if __name__ == '__main__':
    main()
