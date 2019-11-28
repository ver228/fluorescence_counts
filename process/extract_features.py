#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:29:16 2019

@author: avelinojaver
"""
#%%
import pickle
from tqdm import tqdm
from skimage import io 
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
#%%
if __name__ == '__main__':
    
    #img_file = '/Users/avelinojaver/ST_5Plex_ClareV_16118-15_channel_0.tiff'
    #contours_file ='/Users/avelinojaver/Desktop/nuclei_datasets/fluorescence/ST_5Plex_ClareV_16118-15_channel_0.pickle'
    
    img_file = '/well/rittscher/users/wmw296/7plex/step4_Colour_99999_median/ST_5Plex_ClareV_16118-15/ST_5Plex_ClareV_16118-15_channel_0.tiff'
    contours_file = '/users/rittscher/avelino/workspace/segmentation/predictions/fluorescence_multiplexing/BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-5_20191126_094934_adam_lr0.000128_wd0.0_batch128/ST_5Plex_ClareV_16118-15_channel_0.pickle'
    
    contours_file = Path(contours_file)
    save_name = contours_file.parent / (contours_file.stem + '_features.csv')
    
    # I am using skimage because it seems the only one that can read tiff 
    #under the hood it uses `tifffile`, but i runned before into troubles installing it manually
    image = io.imread(img_file)
    
    #It seems that the tif was saved as 3 colors. It is quite a waste of resources since there is fluorescence
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    with open(contours_file, 'rb') as fid:
        cnts = pickle.load(fid)
    
    dat = []
    for cnt in tqdm(cnts):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y+h, x : x + w]
        
        mask = np.zeros(crop.shape, np.uint8)
        corner = np.array((x,y))[None]
        cv2.drawContours(mask, [cnt - corner], 0, 255, -1)
        avg_intensity = cv2.mean(crop, mask = mask)[0]
        #centroid
        cx, cy = cnt.mean(axis=0)
        
        row = (area, avg_intensity, cx, cy)
        dat.append(row)

    df = pd.DataFrame(dat, columns = ['area', 'average_intensity', 'CM_x', 'CM_y'])
    df.to_csv(index = False)