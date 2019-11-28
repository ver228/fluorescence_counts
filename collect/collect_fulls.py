#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:12:04 2019

@author: avelinojaver
"""

from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np

from cell_localization.collect import save_data_single
from collect_in_crops import _read_files, FLUO_INT_THRESH


if __name__ == '__main__':
    #import matplotlib.pylab as plt
    int_threshold = 100

    #root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_train/'
    #save_root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/separated_files/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_train/'
    root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_test/'
    save_root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/separated_files/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_test/'
    
    data = _read_files(root_dir)
    save_root_dir = Path(save_root_dir)
    #%%
    for _key, files in tqdm(data.items()):
        fname = files['images']
        
        img = cv2.imread(str(fname), -1)
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :-1]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_id = fname.stem
        
        
        cnt2check = []
        for ii, fname_mask in enumerate(files['masks']):
            nuclei_id = ii + 1
            
            mask_id = fname_mask.stem
            mm = cv2.imread(str(fname_mask), cv2.IMREAD_ANYDEPTH)
            cnt = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            assert len(cnt) >= 1
            cnt2check += [x.squeeze(1) for x in cnt]
        
        
        centroids = []
        contours = []
        for ii, cnt in enumerate(cnt2check):
            nuclei_id = ii + 1
            
            centroids.append((ii + 1, *np.mean(cnt, axis=0), 1))
            contours += [(nuclei_id, *c) for c in cnt] 
        
        
        #save_name = 
        med = np.median(img)
        if med < FLUO_INT_THRESH:
            img = img[..., 0]
            data_type = 'fluorescence'
        else:
            data_type = 'bright_field'
        
        #offset = 3
        #if med < int_threshold:
        save_name = save_root_dir / data_type / (fname.stem + '.hdf5')
        save_data_single(save_name, img, centroids = centroids,  contours = contours)