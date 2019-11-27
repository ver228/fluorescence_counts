#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:20:05 2019

@author: avelinojaver
"""
import numpy as np
import cv2
import random
  

class Contours2Outputs(object):
    valid_output_types = [ 'contours', 
                           'bboxes', 
                           'centroids', 
                           'point_inside_contour', 
                           'segmentation'
                           ]
    
    
    def __init__(self, output_types = '', channel_lasts = True):
        if isinstance(output_types, str):
            output_types = [output_types]
        
        
        output_types = [x.replace('-', '_') for x in output_types]
        for x in output_types:
            if x not in self.valid_output_types:
                raise ValueError(f'`{x}` is not a valid contour output type.')
        self.output_types = output_types
        
        self.channel_lasts = channel_lasts
        
        
    def compose(self, contours, img_shape):
        outputs = {}
        for out_type in self.output_types:
            dd = getattr(self, out_type)(contours, img_shape)
            assert isinstance(dd, dict)
            outputs.update(dd) 
        return outputs
    
    def __call__(self, image, target):
        img_shape = image.shape[:2] if self.channel_lasts else image.shape[-2:]
        
        target_out = self.compose(target['contours'], img_shape)
        return image, target_out
    
    
    @staticmethod
    def contours(contours, *args, **argkws):
        return {'contours' : contours}
    
    
    @staticmethod
    def centroids(contours, *args, **argkws):
        cms = [np.mean(x, axis=0) for x in contours]
        
        if cms:
            coordinates = np.array(cms, dtype = np.int)
        else:
            coordinates = np.zeros((0, 2), dtype = np.int)
        
        out2 = dict(
                coordinates = coordinates,
                labels = np.ones(coordinates.shape[0])
                )
        return out2
    
    @staticmethod
    def bboxes(contours, *args, **argkws):
        bboxes = [(x[:, 0].min(), x[:, 1].min(), x[:, 0].max(), x[:, 1].max()) for x in contours]
        
        if bboxes:
            bboxes = np.array(bboxes, dtype = np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype = np.int)
        out2 = dict(bboxes = bboxes, 
                    labels = np.ones(bboxes.shape[0])
                    )
        return out2
    
    @staticmethod
    def point_inside_contour(contours, *args, **argkws):
        #I am assuming that most of the time the points will be inside the contour. 
        #I think this is faster than getting all the points by drawing the contour
        points = [] 
        for cnt in contours:
            cnt = cnt.round().astype(np.int)
            xlims = cnt[:, 0].min(), cnt[:, 0].max()
            ylims = cnt[:, 1].min(), cnt[:, 1].max()
            for _ in range(100):
                x = random.randint(*xlims)
                y = random.randint(*ylims)
                
                is_inside = cv2.pointPolygonTest(cnt, (x, y), False)
                if is_inside == 1:
                    break
            else:
                x, y = np.mean(cnt, axis=0)
        
            points.append((x,y))
                
        if points:
            coordinates = np.array(points, dtype = np.int)
        else:
            coordinates = np.zeros((0, 2), dtype = np.int)
        out2 = dict(
                coordinates = coordinates,
                labels = np.ones(coordinates.shape[0])
                )
        return out2
    
    @staticmethod
    def segmentation(contours, img_shape):
        contours_i = [np.floor(x).astype(np.int) for x in contours]
        
        seg_mask = np.zeros(img_shape, dtype = np.uint8)
        
        contours_i = sorted(contours_i, key = cv2.contourArea)
        for cnt in contours_i:
            cv2.drawContours(seg_mask, [cnt], 0, 1, -1)
            
            
        thickness = 1
        for cnt in contours_i:
            cv2.drawContours(seg_mask, [cnt], 0, 2, thickness)
        
        return dict(segmentation_mask = seg_mask.astype(np.int))