#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:15:09 2019

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir/ 'scripts'))

from fluorescence_counts.unet_n2n import ModelWithN2N

from cell_localization.models import get_model
from cell_localization.utils import get_device

from pathlib import Path
import torch
import numpy as np
import cv2
import matplotlib.pylab as plt

if __name__ == '__main__':
    
    #nms_threshold_rel = 0.2
    cuda_id = 0
    device = get_device(cuda_id)
    
    #bn = 'BBBC038-crops+FNone+centroids+noise2noise+roi128_clf+unet-simple+n2n_maxlikelihood_20191115_190453_adam_lr1e-05_wd0.0_batch64'
    #bn = 'BBBC038-crops+sameimages+FNone+centroids+noise2noise+roi128_clf+unet-simple+n2n_maxlikelihood_20191118_161558_adam_lr1e-05_wd0.0_batch64'
    #bn = 'BBBC038-crops+FNone+centroids+noise2noise+roi128_clf+unet-simple+n2n_maxlikelihood_20191118_161558_adam_lr1e-05_wd0.0_batch64'
    
    #bn = 'BBBC038-crops+nobadcrops+FNone+centroids+noise2noise+roi128_clf+unet-simple+n2n_maxlikelihood_20191119_101425_adam_lr1e-05_wd0.0_batch64'
    #bn = 'BBBC038-crops+FNone+centroids+noise2noise+roi128_clf+unet-resnet101+n2n_maxlikelihood_20191119_122514_adam_lr1e-05_wd0.0_batch64'
    bn = 'BBBC038-crops+nobadcrops+FNone+centroids+noise2noise+roi128_clf+unet-resnet101+n2n_maxlikelihood_20191119_141423_adam_lr1e-05_wd0.0_batch64'
    
    results_dir = Path.home() / 'workspace/localization/results/locmax_detection/BBBC038-crops/'
    
    
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_clf+unet-simple_maxlikelihood_20191113_183205_adam_lr0.000128_wd0.0_batch128'
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_clf+unet-resnet101_maxlikelihood_20191119_142057_adam_lr0.000128_wd0.0_batch128'
    #results_dir = Path.home() / 'workspace/localization/results/locmax_detection/BBBC038/fluorescence/BBBC038-fluorescence/'
    
    nms_threshold_rel = 0.05
    
    #model_path = results_dir / bn / 'checkpoint.pth.tar'
    model_path = results_dir / bn / 'model_best.pth.tar'
    assert model_path.exists()
    #
    #%%
    n_ch_in = 1
    n_ch_out = 1
    
    eval_dist = 10
    
    parts = bn.split('_')
    
    loss_type = parts[2]
    model_name = parts[1]
    
    model_type, is_n2n, n2n_type = model_name.partition('+n2n')
    
    model = get_model(model_type, 
                      n_ch_in, 
                      n_ch_out, 
                      loss_type,
                      return_belive_maps = True,
                      nms_threshold_abs = 0.,
                      nms_threshold_rel = nms_threshold_rel
                      )
    if is_n2n:
        model = ModelWithN2N(n_ch_in, model, n2n_return = True)
    
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    #fname = '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'
    fnames = [
            '/Users/avelinojaver/Desktop/h_0_1373_w_12831_14664.jpg',
            '/Users/avelinojaver/Desktop/h_20595_21968_w_0_1833.jpg',
            '/Users/avelinojaver/OneDrive - Nexus365/heba/spheroids/sample_nanog_mastrixplate2_20x/H - 6(fld 24 wv UV - DAPI z 4).tif',
            ]
    #%%
    for fname in fnames:
        
        #ind = 17
        #image, target = val_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32)
        #image = image[:512, -512:]
        #image /= 255
        
        image =  image[800:1300, 900:1400]
        image /= image.max()
        
        image = torch.from_numpy(image[None])
        
        image = image.to(device)
        #target = {k: v.to(device) for k, v in target.items()}
        
        with torch.no_grad():
            outs = model(image[None])
        
        predictions, (xhat, belive_maps) = outs[:2]
        if len(outs) == 3:
            x_n2n = outs[-1][0,0].cpu().numpy()
        
        pred = {k: v.cpu() for k, v in predictions[0].items()}
    
        pred_coords = pred['coordinates'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
            
        #target_coords = target['coordinates'].detach().cpu().numpy()
        #target_labels = target['labels'].detach().cpu().numpy()
        
        
        #TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, target_coords, max_dist = eval_dist, assigment = 'greedy')
        #metrics += TP, FP, FN
        
        has_cell_prob =  belive_maps[0, 0].cpu().detach()
        prob_maps = xhat[0, 0].cpu().detach()
        img = image[0].cpu().numpy()
        
        
        #%%
        figsize = (40, 160)
        #figsize = (20, 80)
        #figsize = (5, 20)
        
        if not is_n2n:
            fig, axs = plt.subplots(1, 4, figsize =figsize, sharex = True, sharey = True)
            
        else:
            fig, axs = plt.subplots(1, 5, figsize =figsize, sharex = True, sharey = True)
            axs[-1].imshow(x_n2n, cmap = 'gray')
        
        #fig, axs = plt.subplots(1, 4, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        axs[2].imshow(prob_maps, vmax = prob_maps.max()*0.5)
        axs[3].imshow(has_cell_prob, vmax = 1.)
        for ax in axs:
            ax.axis('off')
        axs[1].plot(pred_coords[:, 0], pred_coords[:, 1], 'x', color = 'r')
        
        
        #%%
    