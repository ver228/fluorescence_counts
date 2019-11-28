#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:37:48 2019

@author: avelinojaver
"""
import pandas as pd
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_test/'
src_file = '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_test.csv'

root_dir = Path(root_dir)
src_file = Path(src_file)

encoded_pixels = pd.read_csv(src_file)
encoded_pixels_g = encoded_pixels.groupby('ImageId').groups

fnames = [x for x in root_dir.rglob('*.png') if x.parent.name == 'images']

for fname in tqdm(fnames):
    img_src = cv2.imread(str(fname))
    
    image_id = fname.stem
    
    image_annotations = encoded_pixels.loc[encoded_pixels_g[image_id]]
    
    for irow, row in image_annotations.iterrows():
        
        row['EncodedPixels'].split()
        dd = [int(x) for x in row['EncodedPixels'].split()]
        starting_pixels = dd[::2]
        pixels_offsets = dd[1::2]
        
        pixels2add = []
        
        for x, n in zip(starting_pixels, pixels_offsets):
            ini = x -1#it seems pixels are encoded with index starting at 1
            pixels2add += list(range(ini , ini + n )) 
            
        assert img_src.shape[0] == row['Height']
        assert img_src.shape[1] == row['Width']
        
        
        mask = np.zeros(row['Height']*row['Width'], dtype = np.uint8)
        mask[pixels2add] = 255
        mask = mask.reshape((row['Width'], row['Height'])).T
        
        save_name = root_dir / image_id / 'masks' / f'{irow}.png'
        save_name.parent.mkdir(parents = True, exist_ok = True)
        cv2.imwrite(str(save_name), mask)
        
        
        
    
    