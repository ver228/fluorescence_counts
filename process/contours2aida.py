#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:29:16 2019

@author: avelinojaver
"""
#%%
import pickle
import json
from pathlib import Path

def contours2aida(contours, save_file):
    

    color = {"stroke":{"hue":0,"saturation":0.44,"lightness":0.69,"alpha":1}}
    layer_ = {"name": "nuclei",   
              'opacity' : 1,
              "items" : [{"type" : "path", 
                          "color": color ,
                          "closed" : True,
                          "segments" : cnt.tolist()} for cnt in contours]
                  }
    
    out_dict = {"name":"An AIDA project",
                "layers": [layer_]
                }
    
    with open(save_file, 'w') as fid:
        json.dump(out_dict, fid)

if __name__ == '__main__':
    src_file = '/users/rittscher/avelino/workspace/segmentation/predictions/fluorescence_multiplexing/BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-5_20191126_094934_adam_lr0.000128_wd0.0_batch128/ST_5Plex_ClareV_16118-15_channel_0.pickle'
    #src_file = Path('/Users/avelinojaver/Desktop/nuclei_datasets/fluorescence/ST_5Plex_ClareV_16118-15_channel_0.pickle')
    
    
    src_file = Path(src_file)
    save_file = src_file.parent / (src_file.stem + '.json')
    
    with open(src_file, 'rb') as fid:
        contours = pickle.load(fid)

    contours2aida(contours, save_file)
    