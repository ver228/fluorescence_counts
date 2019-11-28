#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:15:09 2019

@author: avelinojaver
"""

from unet_n2n import UNetN2N
from train_n2n import get_criterion
from train import data_args

from cell_localization.flow import CoordFlow
from cell_localization.models import get_mapping_network, model_types
from cell_localization.utils import get_device
from cell_localization.evaluation.localmaxima import score_coordinates

from pathlib import Path
import torch
import numpy as np
import tqdm
import matplotlib.pylab as plt


if __name__ == '__main__':
    
    #nms_threshold_rel = 0.2
    cuda_id = 0
    device = get_device(cuda_id)
    
    
    results_dir = Path.home() / 'workspace/localization/results/locmax_detection/noise2noise/BBBC038-crops'
    #bn = 'BBBC038-crops+FNone+noise2noise+roi64_unet-simple_l1smooth_20191114_090353_adam_lr1e-05_wd0.0_batch256'
    #bn = 'BBBC038-crops+FNone+noise2noise+roi64_unet-n2n_l1smooth_20191114_090634_adam_lr1e-05_wd0.0_batch256'
    #bn = 'BBBC038-crops+FNone+noise2noise+roi64_unet-n2n_l1smooth_20191114_121612_adam_lr1e-05_wd0.0_batch256'
    
    #bn = 'BBBC038-crops+FNone+noise2noise+roi64_unet-n2n_l1smooth_20191114_170940_adam_lr1e-05_wd0.0_batch256'
    bn = 'BBBC038-crops+FNone+noise2noise+roi64_unet-n2n_l1smooth_20191114_222310_adam_lr1e-05_wd0.0_batch256'
    
    
    nms_threshold_rel = 0.05
    
    model_path = results_dir / bn / 'checkpoint-99.pth.tar'
    #model_path = results_dir / bn / 'checkpoint.pth.tar'
    #model_path = results_dir / bn / 'model_best.pth.tar'
    assert model_path.exists()
    #
    #%%
    n_ch_in = 1
    n_ch_out = 1
    
    eval_dist = 10
    
    parts = bn.split('_')
    
    loss_type = parts[2]
    model_name = parts[1]
    
    
 
    model = get_mapping_network(n_ch_in, n_ch_out, **model_types[model_name])
    #model = UNetN2N(n_ch_in, n_ch_out)
    criterion = get_criterion(loss_type)
    state = torch.load(model_path, map_location = 'cpu')
    
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    coord_type = 'noise2noise'
    
    val_dir = data_args['val_dir']
    gen = CoordFlow(val_dir, 
                scale_int = (0, 255.),
                valid_labels = [1],
                is_preloaded = True,
                )
    #%%

    N = len(gen.data_indexes)
    #inds2check = list(range(N))
    inds2check = [3, 10, 12]
    for ind in tqdm.tqdm(inds2check):
        Xin, target = gen.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        
        Xin = Xin[None].to(device)
        #target = {k: v.to(device) for k, v in target.items()}
        
        with torch.no_grad():
            Xout = model(Xin)
        
        
        img = Xin[0, 0].detach().cpu().numpy()
        xout = Xout[0, 0].detach().cpu().numpy()
        
        #figsize = (40, 160)
        #figsize = (20, 80)
        figsize = (10, 20)
        
        fig, axs = plt.subplots(1, 2, figsize = figsize, sharex = True, sharey = True)
        #fig, axs = plt.subplots(1, 4, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(xout, cmap = 'gray')
        
        