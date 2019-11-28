#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:50:20 2019

@author: avelinojaver
"""

import pickle
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    src_file = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/BBBC038_Kaggle_2018_Data_Science_Bowl.p'
    
    save_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/test_bgnd'
    
    src_file = Path(src_file)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)
    
    
    with open(src_file, 'rb') as fid:
        data = pickle.load(fid)
    
    
    min_size = 10
    
    cell_crops = [x for x in data['cell_crops'] if min(x['image'].shape)>=min_size]
    backgrounds = data['backgrounds']
    
    #%%
    for b in tqdm(backgrounds):
        bgnd = b['image']
        img_id = b['img_id']
        
        bgnd = bgnd.astype(np.float32)
        bgnd = bgnd/bgnd.max()
        bgnd = (255*bgnd).astype(np.uint8)
        
        save_name = save_dir / (img_id + '.png')
        cv2.imwrite(str(save_name), bgnd)