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

import pickle

FLUO_INT_THRESH = 100 #lower it will be a flourescent otherwise it is likely to be histology

def _contour2crop(_cnt, _img, offset = 3):
    x,y,w,h = cv2.boundingRect(_cnt)
    x = max(0, x - offset)
    y = max(0, y - offset)
    w += offset
    h += offset
    
    crop = _img[y:y+h, x:x+w].copy()
    
    crop_mask = np.zeros_like(crop)
    cnt_offseted = _cnt.copy()
    cnt_offseted[..., 0] -= x
    cnt_offseted[..., 1] -= y
    cv2.drawContours(crop_mask, [cnt_offseted], 0, 255, -1)
    
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    crop_mask = cv2.dilate(crop_mask, kernel)

    crop = cv2.bitwise_and(crop, crop_mask)
    
    return crop, cnt_offseted, (x,y)


def _read_files(root_dir):
    root_dir = Path(root_dir)
    bad_files = [
            '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80',
            '12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40',
            'adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df',
            ]
    
    
    data = {}
    for fname in root_dir.rglob('*.png'):
        _key = fname.parents[1].name
        if _key in bad_files:
            continue
        
        _type = fname.parent.name
        
        if not _key in data:
            data[_key] = {}
        
        if _type == 'images':
            data[_key][_type] = fname
        elif _type == 'masks':
            if not _type in data[_key]:
                data[_key][_type] = []
            data[_key][_type].append(fname)
        else:
            raise ValueError(f'Type {_type} not recognized.')
    return data

if __name__ == '__main__':
    #import matplotlib.pylab as plt

    root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_train/'
    save_name = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/BBBC038_Kaggle_2018_Data_Science_Bowl_v2.p'
    
    data = _read_files(root_dir)
    
            
    #%%
    
    
    img_shapes = []
    
    
    
    cell_crops = []
    bad_crops = []
    backgrounds = []
    #%%
    min_area_spurious = 20
    for _key, files in tqdm(data.items()):
        #%%
        fname = files['images']
        img_id = fname.stem
        img = cv2.imread(str(fname), cv2.IMREAD_ANYDEPTH)
        
        if np.median(img) > FLUO_INT_THRESH:
            continue
            
        
        is_cell_mask = np.zeros(img.shape[:2], dtype = np.uint8)
        img_contours = []
        for fname_mask in files['masks']:
            mask_id = fname_mask.stem
            
            mm = cv2.imread(str(fname_mask), cv2.IMREAD_ANYDEPTH)
            is_cell_mask |= mm
            cnts = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            
            for cnt in cnts:
                img_contours.append(cnt.squeeze(1))
        
        def _is_in_border(cnt):
            return np.any(cnt <= 1) or np.any(cnt[..., 0] >= img.shape[1]-2) or np.any(cnt[..., 1] >= img.shape[0]-2)
        
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        is_cell_mask = cv2.dilate(is_cell_mask, kernel)
        cnt_clusters = cv2.findContours(is_cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        
        
        bgnd_mask = np.full(img.shape, 255, np.uint8)
        img_crops = []
        for cnt_cluster in cnt_clusters:
            cv2.drawContours(bgnd_mask, [cnt_cluster], 0, 0, -1)
            
            if _is_in_border(cnt_cluster):
                continue
            
            crop, cnt_offset, corner = _contour2crop(cnt_cluster, img)
            corner = np.array(corner)[None]
            
            valid_cnts = [x - corner for x in img_contours if cv2.pointPolygonTest(cnt_cluster, tuple(x[0]), False) >= 0]
            
            crop_data = dict(
                    img_id = img_id,
                    mask_id = mask_id,
                    image = crop,
                    contours = valid_cnts
                    )
            img_crops.append(crop_data)
            
        
        
        if not img_crops:
            continue
        cell_crops += img_crops
        dimmer_nuclei = min([np.median(x['image']) for x in img_crops])
        #%%
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        bgnd_mask = cv2.erode(bgnd_mask, kernel)
        bgnd = img.copy()
        bgnd = cv2.bitwise_and(bgnd, bgnd_mask)
        
        #%%
        _, spurious_mask = cv2.threshold(bgnd, dimmer_nuclei, 255, cv2.THRESH_BINARY)
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        spurious_mask = cv2.morphologyEx(spurious_mask, cv2.MORPH_CLOSE, kernel)
        
        spurious_cnts = cv2.findContours(spurious_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        
        max_size = bgnd.size//8
        _is_valid = lambda x : (x > min_area_spurious) & (x < max_size)
        
        spurious_cnts = [x.squeeze(1) for x in spurious_cnts if _is_valid(cv2.contourArea(x))]
        img_spurious_crops = []
        for cnt in spurious_cnts:
            
            crop, cnt_offset, _ = _contour2crop(cnt, bgnd)
            crop_data = dict(
                    img_id = img_id,
                    image = crop,
                    contours = [cnt_offset]
                    )
            
            img_spurious_crops.append(crop_data)    
            cv2.drawContours(bgnd, [cnt], 0, 0, -1)
            
        bad_crops += img_spurious_crops
            
        bngd_data = dict(
                img_id = img_id,
                image = bgnd
                )
        
        backgrounds.append(bngd_data)
        
            
    #%%
    
    data = {'cell_crops' : cell_crops,
            'bad_crops' : bad_crops,
            'backgrounds' : backgrounds
            }
    
    with open(save_name, 'wb') as fid:
        pickle.dump(data, fid)
    