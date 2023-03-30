import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D, Reshape, MaxPooling2D, MaxPooling3D, Conv2DTranspose, GlobalAveragePooling2D, GlobalAveragePooling3D, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time

result = np.load('IP_namd.npy',allow_pickle=True)
data_path = os.path.join(os.getcwd(),'data')
gt_mat = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))
gt=gt_mat['indian_pines_gt']

#result = np.reshape(result,(145,145,16))

dataset = 'IP'
test_ratio = 0.95
windowSize_2D = 25
windowSize_3D = 25
X_2D=result
y_2D=gt


# datagen = ImageDataGenerator(
#     vertical_flip=True,
#     horizontal_flip=True)


def loadData(name):
    data_path = os.path.join(os.getcwd(),'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    
    return data, labels

def splitTrainTestSet(X, y, testRatio, randomState=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,stratify=y)
    return X_train, X_test, y_train, y_test

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

X, y = loadData(dataset)
X,pca = applyPCA(X,numComponents=16)
X, y = createImageCubes(X, y, windowSize=windowSize_3D)
X_2D, y_2D = createImageCubes(X_2D, y_2D, windowSize=windowSize_2D)
Xtrain_3D, Xtest_3D, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
Xtrain_2D, Xtest_2D, ytrain_2D, ytest_2D = splitTrainTestSet(X_2D, y_2D, test_ratio)


Xtrain_3D = Xtrain_3D.reshape(-1, windowSize_3D, windowSize_3D, 16)
Xtrain_2D = Xtrain_2D.reshape(-1, windowSize_2D, windowSize_2D, 16)
ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)


#-----------------------------2D-----------------------------------------
in_2D = Input((25, 25, 16))
model_2D = Conv2D(16, kernel_size=(3,3) , strides=(1,1), activation='relu', padding='valid')(in_2D)
#model_2D = MaxPooling2D(pool_size=(3,3) , strides=(1,1))(model_2D)
model_2D = Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = MaxPooling2D(pool_size=(2,2) , strides=(1,1), padding='same')(model_2D)
model_2D = Conv2D(64, kernel_size=(5,5), strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = Conv2DTranspose(64, kernel_size=(5,5) , strides=(1,1), activation='relu', padding='valid')(model_2D)
model_2D = GlobalAveragePooling2D()(model_2D)


#------------------------------3D-----------------------------------------
in_3D = Input((25, 25, 16, 1))
model_3D = Conv3D(8, kernel_size=(3,3,7), strides=(1,1,1), padding='same',activation='relu')(in_3D)
model_3D = Conv3D(16, kernel_size=(3,3,5), strides=(1,1,1), padding='same',activation='relu')(model_3D)
model_3D = MaxPooling3D(pool_size=(3,3,3) , strides=(1,1,1), padding='same')(model_3D)
model_3D = Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1),  padding='same', activation='relu')(model_3D)
model_3D = GlobalAveragePooling3D()(model_3D)


merged = Concatenate()([model_2D, model_3D])
output = Dense(1024, activation='relu')(merged)
output = Dropout(0.25)(output)
output = Dense(512, activation='relu')(output)
output = Dropout(0.25)(output)
output = Dense(128, activation='relu')(output)
output = Dropout(0.25)(output)
output = Dense(16, activation='softmax')(output)

model_combined = Model(inputs=[in_2D, in_3D], outputs=[output])
adam = Adam(lr=0.001, decay=1e-06)
model_combined.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model_combined.summary()
start = time.time()

# history = model_combined.fit(
#         datagen.flow([Xtrain_2D, Xtrain_3D], ytrain, batch_size=100),
#         validation_data = datagen.flow([Xtrain_2D, Xtrain_3D], ytrain, batch_size=100), epochs = 200)



history = model_combined.fit([Xtrain_2D, Xtrain_3D], ytrain, validation_split=0.05, epochs=300, batch_size=100)
print("Total training time: ", time.time() - start, "seconds")


loss, accuracy = model_combined.evaluate([Xtest_2D, Xtest_3D] , ytest)
#print('Test:')
print('loss: ',loss)
print('accuracy: ',accuracy)

model_combined.save('hybrid_model.hdf5')


#%%

# load best weights
model_combined.load_weights("hybrid_model.hdf5")
Y_pred_test = model_combined.predict([Xtest_2D, Xtest_3D])
y_pred_test = np.argmax(Y_pred_test, axis=1)

classification = classification_report(np.argmax(ytest, axis=1), y_pred_test)
print(classification)

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (Xtest_2D, Xtest_3D, y_test, name):
    start = time.time()
    Y_pred = model_combined.predict([Xtest_2D, Xtest_3D])
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
    
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model_combined.evaluate([Xtest_2D, Xtest_3D], y_test, batch_size=100)
    Test_Loss =  score[0]*100
    Test_accuracy = score[1]*100
    return classification, confusion, Test_Loss, Test_accuracy, oa*100, each_acc*100, aa*100, kappa*100


classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest_2D, Xtest_3D, ytest, dataset)
classification = str(classification)
confusion = str(confusion)
file_name = "combined_classification_report.txt"

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
    
