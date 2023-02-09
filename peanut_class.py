# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:44:57 2021

@author: Joe
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, segmentation,color
class peanut:
#    @staticmethod
    def test_peanut_roi(self,img,mask):
        try: 
            peanut_roi = []
            peanut_coordinate = []
            img_la_overlay = []
            cleared = mask.copy()  #複製
            
            segmentation.clear_border(cleared)
            label_image =measure.label(cleared)  #連續區域標記
            borders = np.logical_xor(mask, cleared) #異物
            label_image[borders] = -1
            image_label_overlay =color.label2rgb(label_image, image=img) #不同標記用不同顏色顯示
            x = 2
            for region in measure.regionprops(label_image):      
                #忽略小區域
                if region.area <2000:
                    continue
                #ROI
                minr, minc, maxr, maxc = region.bbox      
                peanut_roi.append(img[minr-x:maxr+x,minc-x:maxc+x,:])
                peanut_coordinate.append(region.bbox)
                
                
            img_la_overlay.append(image_label_overlay)
        except:
            print('ROI error')
        else:
            return peanut_roi,peanut_coordinate,img_la_overlay
    
##############################################################################
##############################################################################
    def avg_peanuts(self,im, label):
        im_2d = np.reshape(im, [-1, im.shape[2]])
        label_2d = np.reshape(label, -1)
        no_ls = []
        
        '''將不要的標籤編號放到no_list'''
        for i in range(label.max() + 1):
            if i == 0 or np.count_nonzero(label == i) < 2000:  #0是背景我不要，點數太少我也不要
                no_ls.append(i)
        
        '''將一張圖每個標籤的光譜訊號分別加到各標籤的array後再平均'''
        '''arr_peanuts的大小是[標籤數, 波段數]'''
        '''peanutslabel_ls是可能為花生的標籤編號list'''
        '''假設某點的標籤是2，那將這個點的光譜訊號放到arr_peanuts的大小是[2]'''      
        arr_peanuts = np.zeros([label.max() + 1, im.shape[2]])    
        peanutslabel_ls = list(set(range(label.max() + 1)) - set(no_ls))
        for index in peanutslabel_ls:
            arr_peanuts[index] = im_2d[np.where(label_2d == index)].sum(0, dtype = 'float64')
    
        '''將arr_peanuts分別除以各標籤的點數量'''
        for i in range(label.max() + 1):
            count = np.count_nonzero(label == i)
            if count > 0:
                arr_peanuts[i] = arr_peanuts[i]/np.count_nonzero(label == i)
        
        arr_peanuts = np.delete(arr_peanuts, no_ls, axis = 0)  #刪除非花生的列
        
        return arr_peanuts, peanutslabel_ls
    
#############################################################################
#############################################################################
    '''
    花生分類並畫圖
    輸入:  arr_peanuts: 每顆花生的平均光譜訊號，2d-array
           peanutslabel_ls: 每顆花生的標籤編號，1d-list
           label: label圖，同一顆花生用相同數字表示，2d-array
    '''
    def peanuts_classification(self,peanuts_pre, peanutslabel_ls, label,num):
        y_hat = peanuts_pre
        print(y_hat.shape)
        label_copy = label.copy()
        label_copy[label != peanutslabel_ls] = 0
        for index, value in enumerate(peanutslabel_ls):
            if y_hat[index] == 0:  #預測為好(0)
                label_copy[label == value] = 200  #200是米色
            elif y_hat[index] == 1:  #預測為不好(1)
                label_copy[label == value] = 100  #100是洋紅色
                
        label_copy[0, 0] = 200
        plt.figure()
        plt.axis("off")
        plt.imshow(label_copy, cmap = 'magma')
#        plt.savefig(f"D:/peanut_classify/圖/3DCNN/全波段圖/{num}_RT.png")
        return label_copy
    
    