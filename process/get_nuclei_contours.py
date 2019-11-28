#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:31:14 2019

@author: avelinojaver
"""
#import sys
#from pathlib import Path 
#root_dir = Path(__file__).resolve().parents[1]
#sys.path.append(str(root_dir))
#
#from fluorescent_counts.unet_n2n import ModelWithN2N #i should incorporate this function at some point the cell_localization module

from cell_localization.models import get_model
from cell_localization.utils import get_device
from cell_localization.evaluation import segmentation2contours

from pathlib import Path
from skimage import io 
import cv2
import numpy as np
from tqdm import tqdm
import torch
import pickle

def _load_data(src_file, 
               crop_size = (2048, 2048), 
               border_offset = 32  
               ):
    # I am using skimage because it seems the only one that can read tiff 
    #under the hood it uses `tifffile`, but i runned before into troubles installing it manually
    image = io.imread(src_file)
    
    #It seems that the tif was saved as 3 colors. It is quite a waste of resources since there is fluorescence
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    data2analyze = []
    for ii in range(0, image.shape[0], crop_size[1] - border_offset):
        for jj in range(0, image.shape[1], crop_size[1] - border_offset):
            corner = (jj, ii)
            crop = image[ii:ii+crop_size[0], jj:jj+crop_size[1]] #since this is copied by reference, i can reshape the data here
    
            data2analyze.append((corner, crop))
            
    return data2analyze

def _load_model(model_path):
    model_path = Path(model_path)
    bn = model_path.parent.name # I need teh path of the parent directory since it contains the info about the model
    
    n_ch_in = 1
    n_ch_out = 3
    parts = bn.split('_')
    
    loss_type = parts[2]
    model_name = parts[1]
    
    model_type, is_n2n, n2n_type = model_name.partition('+n2n')
    
    model = get_model(model_type, 
                      n_ch_in, 
                      n_ch_out, 
                      loss_type
                      )
    #if is_n2n:
    #    model = ModelWithN2N(n_ch_in, model, n2n_return = True)
    
    state = torch.load(model_path, map_location = 'cpu')
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model

#%%
def _bboxes_overlap_fraction(box_a,box_b):
    def get_area(box):
        return (box[:,2]-box[:,0]) * (box[:,3] - box[:,1])
    
    #this is a fast method to find what boxes have some intersection  
    min_xy = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    max_xy = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    
    #negative value here means no inteserction
    inter = (max_xy-min_xy).clip(min=0)
    inter = inter[...,0] * inter[...,1]
    
    target_area = get_area(box_a)[:,None] #+ get_area(box_b)[None] - inter
    
    overlap = inter/target_area
    
    return overlap
#%%
def _filter_overlaping_contours(contours, overlap_threshold = 0.8):
    
    bboxes = np.array([cv2.boundingRect(x) for x in contours])
    #xywh to xyxy
    bboxes[:, 2:] += bboxes[:, :2]
    
    inds2remove = []
    for target_bbox in bboxes:
        #I will only keep the contour with the larger area from the overlaps. I am skewing the data to mergers
        overlaps = _bboxes_overlap_fraction(target_bbox[None], bboxes) # I need to iterate otherwise the memory comsuption will be too large
        match = np.where(overlaps[0]> overlap_threshold)[0]
        if match.size > 1:
            match = sorted(match, key = lambda x : cv2.contourArea(contours[x]))
            inds2remove += match[:-1]
        
    inds2keep = set(range(len(contours))) - set(inds2remove)
    
    contours2keep = [contours[i] for i in inds2keep]
    return contours2keep
#%%

if __name__ == '__main__':
#    src_file = '/Users/avelinojaver/ST_5Plex_ClareV_16118-15_channel_0.tiff'
#    model_path = Path.home() / 'Desktop/nuclei_datasets/BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-10_20191120_213741_adam_lr0.000128_wd0.0_batch128/checkpoint-199.pth.tar'
#    save_dir = '/Users/avelinojaver/fluorescence_multiplexing'
#    crop_size = (256, 256)
#    
    
    src_file = '/well/rittscher/users/wmw296/7plex/step4_Colour_99999_median/ST_5Plex_ClareV_16118-15/ST_5Plex_ClareV_16118-15_channel_0.tiff'
    model_path = '/users/rittscher/avelino/workspace/localization/results/segmentation/BBBC038/fluorescence/BBBC038-fluorescence/BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-5_20191126_094934_adam_lr0.000128_wd0.0_batch128/model_best.pth.tar'
    save_dir = '/users/rittscher/avelino/workspace/segmentation/predictions/fluorescence_multiplexing'
    crop_size = (2048, 2048)
    
    border_offset = 32 #i will iterarte overlaping boerders in order to resample contours in the border
    cuda_id = 0
    
    
    model_path = Path(model_path)
    src_file = Path(src_file)
    save_dir = Path(save_dir)
    
    save_name = Path(save_dir) / model_path.parent.name / (src_file.stem + '.pickle')
    save_name.parent.mkdir(parents = True, exist_ok = True)
    
    data2analyze = _load_data(src_file, border_offset = border_offset, crop_size = crop_size)
    model = _load_model(model_path)
    
    device = get_device(cuda_id)
    model = model.to(device)
    
    def _get_cnt_region(x):
        is_y_middle = False
        if (x[..., 1] < border_offset).any():
            y_range = (0, border_offset)
        elif (x[..., 1] >= crop_size[1] - border_offset).any():
            y_range = ((crop_size[1] - border_offset), crop_size[1])
        else:
            y_range =  (border_offset, (crop_size[1] - border_offset))
            is_y_middle = True
        
        
        is_x_middle = False
        if (x[..., 0] < border_offset).any():
            x_range = (0, border_offset)
        elif (x[..., 0] >= crop_size[0] - border_offset).any():
            x_range = ((crop_size[0] - border_offset), crop_size[0])
        else:
            x_range =  (border_offset, (crop_size[0] - border_offset))
            is_x_middle = True
        
        
        is_middle = is_x_middle & is_y_middle
        
        return (*x_range, *y_range), is_middle
    
    contours_in_center = []
    contours_in_border = {}
    for corner, image in tqdm(data2analyze):
        if not image.any(): #empty patch nothing to do here
            continue
        
        with torch.no_grad():
            X = torch.from_numpy(image[None, None]).float()
            X /= 255.
            X = X.to(device)
            
            predictions = model(X)
            labels = predictions[0].cpu().detach().numpy()
        
        #let's ignore contours that touch the border
        cell_contours = segmentation2contours(labels == 1, chain_approx = True)
        def _is_in_border(cnt):
            return (cnt <= 1).any() or (cnt[..., 0] >= labels.shape[1]-2).any() or (cnt[..., 1] >= labels.shape[0]-2).any()
        
        corner = np.array(corner).astype(np.int32)
        
        
        cell_contours = [x for x in cell_contours if not _is_in_border(x)]
        
        for cnt in cell_contours:
            region, is_centre = _get_cnt_region(cnt)
            cnt = (cnt + corner[None]).astype(np.int32)
            if is_centre:
                contours_in_center.append(cnt)
            else: 
                x1, x2, y1, y2 =  region
                key = x1 + corner[0], x2 + corner[0], y1 + corner[1], y2 + corner[1]
                
                
                if not key in contours_in_border:
                    contours_in_border[key] = []
                contours_in_border[key].append(cnt)    
    
    #remove overlaping contours
    cnts_filtered = []
    for cnts in contours_in_border.values():
        cnts_filtered += _filter_overlaping_contours(cnts)
        
    all_contours = cnts_filtered + contours_in_center
    
    # save data
    with open(save_name, 'wb') as fid:
        pickle.dump(all_contours, fid)
        #pickle.dump([contours_in_border, contours_in_center], fid)
        