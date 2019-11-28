#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:38:56 2019

@author: avelinojaver
"""

from cell_localization.flow.transforms import ToTensor, Compose

class ToTensor(object):
    def __call__(self, image, target):
        
        image = torch.from_numpy(image).float()
        
        if 'coordinates' in target:
            target['coordinates'] = torch.from_numpy(target['coordinates']).long()
            target['labels'] = torch.from_numpy(target['labels']).long()
        
        if 'mask' in target:
            target['mask'] = torch.from_numpy(target['mask']).float()
        
        return image, target