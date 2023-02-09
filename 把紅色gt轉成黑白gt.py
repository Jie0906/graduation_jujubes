# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:51:09 2021

@author: Yuchi
"""
from spectral.io import envi
import numpy as np
import matplotlib.pyplot as plt

def 存圖(圖片, 檔名 = 'res'):
    height, width = 圖片.shape
    fig = plt.figure('res')
    plt.axis('off')
    fig.set_size_inches(width/100.0,height/100.0)  #輸出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0, wspace=0)
    plt.margins(0,0)
    plt.imshow(圖片, 'gray')
    plt.show()
    plt.savefig(f'{檔名}.jpeg', pad_inches=0.0)
    
def 存npy(圖片, 檔名 = 'res'):
    np.save(f'{檔名}.npy', 圖片)



root = r'C:\Users\user\Desktop\res'

gt = plt.imread(root + '.jpg')
plt.imshow(gt)
re_gt = gt.reshape([-1, 3])
g = np.zeros(re_gt.shape[0])

for i in range(re_gt.shape[0]):
    if re_gt[i, 0] > 200 and re_gt[i, 1] < 200:
        g[i] = 1
        
plt.figure()
plt.imshow(g.reshape(gt.shape[:-1]))

存圖(g.reshape(gt.shape[:-1]), 'gt')
存npy(g.reshape(gt.shape[:-1]), 'gt')