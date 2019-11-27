#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:38:12 2019

@author: avelinojaver
"""

import numpy as np

class OverlapsTracker():
    def __init__(self, image_shape, max_overlap, null_value):
        
        self.max_overlap = max_overlap
        self.null_value = null_value
        self.overlap_mask = np.zeros(image_shape, np.int32)
        self.bbox_placed = []
        self.crops_placed = []
        
        
    def add(self, xi, yi, crop):
        if crop.ndim == 3: # I do not need the three dimensions to track locations
            crop = crop[..., 0]
        
        img_size = crop.shape
        crop_bw = crop != self.null_value
        
        
        xf = xi + img_size[0]
        yf = yi + img_size[1]
        
        rr = self.overlap_mask[xi:xf, yi:yf]
        overlap_frac = np.mean(rr>0)
        
        #check the fraction of pixels on the new data that will be cover by the previous data
        if overlap_frac > self.max_overlap:
            return False
        
        #check the fraction of pixels in each of the previous crops that will be cover with the new data.
        #I want to speed this up by checking only the bounding boxes of previously placed objects
        bbox = (xi, yi, xf, yf)
        if len(self.bbox_placed):
            overlaps_frac, intersect_coords = self.bbox_overlaps(bbox, self.bbox_placed)
            bad_ = overlaps_frac > self.max_overlap
            
            #if i find a box that has a large overlap then i want to refine the predictions
            if np.any(bad_):
                #i will refine this estimate at the pixel level rather than to the bbox level
                inds2check, = np.where(bad_)
                for ind in inds2check:
                    crop_placed = self.crops_placed[ind]
                    original_area = crop_placed.sum()
                    if original_area == 0:
                        continue
                    
                    bbox_placed = self.bbox_placed[ind]
                    bbox_intersect= intersect_coords[ind]
                    
                    x_bi, x_bf = bbox_intersect[[0, 2]] - bbox_placed[0]
                    assert x_bi >= 0
                    y_bi, y_bf = bbox_intersect[[1, 3]]  - bbox_placed[1]
                    assert y_bi >= 0
                    
                    x_ci, x_cf = bbox_intersect[[0, 2]] - bbox[0]
                    assert x_ci >= 0
                    y_ci, y_cf = bbox_intersect[[1, 3]]  - bbox[1]
                    assert y_ci >= 0
                    
                    
                    prev_area = crop_placed[x_bi:x_bf, y_bi:y_bf]
                    next_area = crop_bw[x_ci:x_cf, y_ci:y_cf]
                    
                    
                    intesected_area = (prev_area & next_area).sum()
                    pix_overlap_frac = intesected_area/original_area
                    
                    assert pix_overlap_frac <= 1.
                        
                    #print(original_area, intesercted_area, pix_overlap_frac)
                    if pix_overlap_frac > self.max_overlap: 
                        return False
                    
        
        
        self.bbox_placed.append(bbox)
        self.crops_placed.append(crop_bw)
        
        curr_ind = len(self.bbox_placed)
        self.overlap_mask[xi:xf, yi:yf][crop_bw] = curr_ind
        
        return True
    
    def bbox_overlaps(self, bbox_new, bbox_placed):
        """
        get bonding max overlaps
        bbox_new: [x1, y1, x2, y2]
        bbox_placed: [[x1, y1, x2, y2]]
        """
        
        bbox_new = np.array(bbox_new)
        bbox_placed  = np.array(bbox_placed)
        
        areas = (bbox_placed[:, 2] - bbox_placed[:, 0] + 1) * (bbox_placed[:, 3] - bbox_placed[:, 1] + 1)
        
        
        xx1 = np.maximum(bbox_new[0], bbox_placed[:, 0])
        yy1 = np.maximum(bbox_new[1], bbox_placed[:, 1])
        xx2 = np.minimum(bbox_new[2], bbox_placed[:, 2])
        yy2 = np.minimum(bbox_new[3], bbox_placed[:, 3])
            
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        overlaps = inter/areas
        
        intersect_coords = np.stack((xx1, yy1, xx2, yy2)).T
        
        return overlaps, intersect_coords