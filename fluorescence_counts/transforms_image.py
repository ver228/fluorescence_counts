#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:18:11 2019

@author: avelinojaver
"""

import cv2
import random
import numpy as np

class AddRandomRectangle():
    def __init__(self, 
                 rect_rel_size = (0.25, 1.), 
                 rect_int_range = (0.05, 0.25),
                 img_shape = (128, 128),
                 prob = 0.5
                 
                 ):
        self.rect_rel_size = rect_rel_size
        self.rect_int_range = rect_int_range
        self.img_shape = [int(x) for x in img_shape]
        self.prob = prob
        
    
    def __call__(self, image, target):
        
         
        if random.random() < self.prob:
            
            box_size = []
            for s in self.img_shape:
                b = random.randint(int(self.rect_rel_size[0]*s), int(self.rect_rel_size[1]*s)) 
                box_size.append(b)
            
            corner = [random.randint(0, s-1) for s in self.img_shape]
            angle = random.randint(-180, 180)
            int_level = random.uniform(0.1, 0.5)
            
            box = cv2.boxPoints((corner, box_size, angle))
            box = np.int0(box)
            
            shape_noise = np.zeros(self.img_shape, np.float32)
            cv2.drawContours(shape_noise, [box], 0, int_level, -1)
            
            shape_noise = cv2.GaussianBlur(shape_noise, (25, 25), 0)
            image = image + shape_noise
            
        return image, target


class RandomN2NTransform():
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
             image = self.apply(image)
             if 'n2n_target' in target:
                 target['n2n_target'] = self.apply(target['n2n_target'])
        
        return image, target
        
    
    def apply(self, img):
        return img
        
class RandomBlur(RandomN2NTransform):
    def __init__(self, 
                 prob = 0.5,
                 kernel_size = 11
                 ):
        super().__init__(prob = prob)
        self.kernel_size = (kernel_size, kernel_size)
        
    def apply(self, img):
        return cv2.GaussianBlur(img, self.kernel_size, 0)
        
class AddGaussNoise(RandomN2NTransform):
    def __init__(self, gauss_scale = (0.01, 0.1),  prob = 0.33):
        super().__init__(prob = prob)
        self.gauss_scale = gauss_scale
        
    def apply(self, img):
        gauss_scale = np.random.uniform(0.01, 0.1)
        img_noisy = img + np.random.normal(0, gauss_scale, size = img.shape).astype(np.float32)
        return np.clip(img_noisy, 0, 1.)
       


class ConvertToPoissonNoise(RandomN2NTransform):
    def __init__(self, 
                 int_scale = (0, 255), 
                 prob = 0.33
                 
                 ):
        super().__init__(prob = prob)
        self._scale = int_scale[1] - int_scale[0]

    def apply(self, img):
        
        #TODO there shouldn't be any nan here, but it some situation it does happens... this is a quick fix
        return np.random.poisson(img*self._scale).astype(np.float32)/self._scale
        