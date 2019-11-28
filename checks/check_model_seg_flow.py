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

from cell_localization.models import get_model
from cell_localization.utils import get_device
from cell_localization.flow import FlowCellSegmentation
from cell_localization.evaluation import get_masks_metrics, get_IoU_best_match


from pathlib import Path
import torch
import numpy as np
import tqdm
import matplotlib.pylab as plt

from train import data_args
from fluorescence_counts.unet_n2n import ModelWithN2N

#%%
if __name__ == '__main__':
    
    #nms_threshold_rel = 0.2
    cuda_id = 0
    device = get_device(cuda_id)
    bn = 'BBBC038-crops+FNone+segmentation+noise2noise+roi128_seg+unet-simple+n2n_crossentropy+W1-1-10_20191123_205458_adam_lr1e-05_wd0.0_batch64'
    bn = 'BBBC038-crops+sameimages+nobadcrops+FNone+segmentation+noise2noise+roi128_seg+unet-simple+n2n_crossentropy+W1-1-10_20191123_205432_adam_lr1e-05_wd0.0_batch64'
    
    #bn = 'BBBC038-crops+FNone+segmentation+noise2noise+roi128_seg+unet-resnet101+n2n_crossentropy+W1-1-10_20191124_184034_adam_lr1e-05_wd0.0_batch48'
    bn = 'BBBC038-crops+sameimages+nobadcrops+FNone+segmentation+noise2noise+roi128_seg+unet-resnet101+n2n_crossentropy+W1-1-10_20191124_183746_adam_lr1e-05_wd0.0_batch64'
    #bn = 'BBBC038-crops+sameimages+nobadcrops+FNone+segmentation+noise2noise+roi128_seg+unet-resnet101+n2n_crossentropy+W1-1-5_20191125_131232_adam_lr1e-05_wd0.0_batch64'
    #bn = 'BBBC038-crops+sameimages+nobadcrops+FNone+segmentation+noise2noise+roi128_seg+unet-resnet101+n2n_WCE_20191125_154036_adam_lr1e-05_wd0.0_batch64'
    results_dir = Path.home() / 'workspace/localization/results/segmentation/BBBC038-crops/'
    
    bn = 'BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-5_20191126_094934_adam_lr0.000128_wd0.0_batch128'
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy_20191120_170207_adam_lr0.000128_wd0.0_batch128'
    
    results_dir = Path.home() / 'workspace/localization/results/segmentation/BBBC038/fluorescence/BBBC038-fluorescence/'
    
    
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-10_20191120_213741_adam_lr0.000128_wd0.0_batch128'
    #results_dir = Path.home() / 'Desktop/nuclei_datasets/'
    
    #model_path = results_dir / bn / 'checkpoint.pth.tar'
    model_path = results_dir / bn / 'model_best.pth.tar'
    assert model_path.exists()
    #
    #%%
    n_ch_in = 1
    n_ch_out = 3
    
    eval_dist = 10
    
    parts = bn.split('_')
    
    loss_type = parts[2]
    model_name = parts[1]
    
    model_type, is_n2n, n2n_type = model_name.partition('+n2n')
    
    
    argkws = dict()
    if 'resnet' in model_type:
        argkws = dict(load_pretrained = False)

    
    model = get_model(model_type, 
                      n_ch_in, 
                      n_ch_out, 
                      loss_type,
                      return_belive_maps = True,
                      **argkws
                      )
    
    
    if is_n2n:
        model = ModelWithN2N(n_ch_in, model, n2n_return = True)
    
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    val_dir = data_args['val_dir']
    gen = FlowCellSegmentation(val_dir, 
                scale_int = (0, 255.),
                valid_labels = [1],
                is_preloaded = True,
                )
    #%%
    metrics = np.zeros(5)
    
    N = len(gen.data_indexes)
    inds2check = list(range(N))
    #inds2check = [1, 3, 10, 20]
    
    for ind in tqdm.tqdm(inds2check):
        
        image, target = gen.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        
        #image /= image.max()
        
        image = image.to(device)
        #target = {k: v.to(device) for k, v in target.items()}
        
        with torch.no_grad():
            outs = model(image[None])
        
        predictions, xhat = outs[:2]
        if len(outs) == 3:
            x_n2n = outs[-1][0,0].cpu().numpy()
        
        
        pred_segmentation = predictions[0].cpu().detach().numpy()
        
        prob_maps = torch.softmax(xhat, dim=1)
        prob_maps = np.rollaxis(prob_maps[0].cpu().detach().numpy(), 0, 3)
        img = image[0].cpu().numpy()
        #%%
        
        true_cells_mask = (target['segmentation_mask']==1).cpu().numpy().astype(np.uint8)
        pred_cells_mask = (pred_segmentation == 1).astype(np.uint8)
        
        
        pred_coords, target_coords, IoU, agg_inter, agg_union = get_masks_metrics(true_cells_mask, pred_cells_mask)
        TP, FP, FN, pred_ind, true_ind = get_IoU_best_match(IoU)
        
        
        metrics += (TP, FP, FN, agg_inter, agg_union) 
        
        
        #figsize = (40, 160)
        #figsize = (20, 80)
        #figsize = (5, 20)
        figsize = None
        
        if not is_n2n:
            fig, axs = plt.subplots(1, 4, figsize =figsize, sharex = True, sharey = True)
            
        else:
            fig, axs = plt.subplots(1, 5, figsize =figsize, sharex = True, sharey = True)
            axs[2].imshow(x_n2n, cmap = 'gray')
        
        #fig, axs = plt.subplots(1, 4, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        axs[-1].imshow(pred_segmentation)
        axs[-2].imshow(prob_maps, vmax = prob_maps.max()*0.5)
        for ax in axs:
            ax.axis('off')
        if not pred_ind.size:
            axs[1].plot(pred_coords[:, 0], pred_coords[:, 1], 'x', color = 'r')
            axs[1].plot(target_coords[:, 0], target_coords[:, 1], '.', color = 'r')
            
        else:
            good = np.zeros(pred_coords.shape[0], np.bool)
            good[pred_ind] = True
            pred_bad = pred_coords[~good]
            
            good = np.zeros(target_coords.shape[0], np.bool)
            good[true_ind] = True
            target_bad = target_coords[~good]
            
            axs[1].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
            axs[1].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
            axs[1].plot(pred_coords[pred_ind, 0], pred_coords[pred_ind, 1], 'o', color='g')
        
    #%%
    print(model_path)
    TP, FP, FN,  agg_inter, agg_union = metrics
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    AJI = agg_inter/agg_union
    
    score_str = f'P : {P:.3}, R : {R:.3}, F1 : {F1:.3}, AJI : {AJI:.3}'
    
    print(score_str)
        