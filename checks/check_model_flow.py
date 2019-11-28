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
from cell_localization.evaluation.localmaxima import score_coordinates
from cell_localization.flow import CoordFlow

from pathlib import Path
import torch
import numpy as np
import tqdm
import matplotlib.pylab as plt


from train import flow_args, load_data, data_args
from fluorescence_counts.flow import FluoMergedFlowTest
from fluorescence_counts.unet_n2n import ModelWithN2N

if __name__ == '__main__':
    
    #nms_threshold_rel = 0.2
    cuda_id = 0
    device = get_device(cuda_id)
    
    
    results_dir = Path.home() / 'workspace/localization/results/locmax_detection/BBBC038-crops/'
    bn = 'BBBC038-crops+FNone+centroids+noise2noise+roi128_clf+unet-resnet101+n2n_maxlikelihood_20191124_012218_adam_lr1e-05_wd0.0_batch64'
    bn = 'BBBC038-crops+sameimages+nobadcrops+FNone+centroids+noise2noise+roi128_clf+unet-resnet101+n2n_maxlikelihood_20191124_184827_adam_lr1e-05_wd0.0_batch64'
    
    #results_dir = Path.home() / 'workspace/localization/results/locmax_detection/BBBC038/fluorescence/BBBC038-fluorescence/'
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_clf+unet-simple_maxlikelihood_20191113_183205_adam_lr0.000128_wd0.0_batch128'
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_clf+unet-simple_maxlikelihood_20191113_183205_adam_lr0.000128_wd0.0_batch128'
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_clf+unet-simple_maxlikelihood_20191116_095821_adam_lr0.000128_wd0.0_batch128'
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_clf+unet-resnet101_maxlikelihood_20191119_142057_adam_lr0.000128_wd0.0_batch128'
    
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
    
    
    argkws = dict()
    if 'resnet' in model_type:
        argkws = dict(load_pretrained = False)

    
    model = get_model(model_type, 
                      n_ch_in, 
                      n_ch_out, 
                      loss_type,
                      return_belive_maps = True,
                      nms_threshold_abs = 0.,
                      nms_threshold_rel = nms_threshold_rel,
                      **argkws
                      )
    
    
    if is_n2n:
        model = ModelWithN2N(n_ch_in, model, n2n_return = True)
    
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    
    coord_type = 'centroids'
    #src_file = data_args['src_file']
    #cell_crops, bad_crops, backgrounds = load_data(src_file)    
    #del flow_args['n_cells_per_crop']    
#    gen = FluoMergedFlowTest(cell_crops = cell_crops,
#                         bad_crops = bad_crops,
#                         backgrounds = backgrounds,
#                         epoch_size = 25,
#                         output_type = coord_type,
#                         img_size = (128, 128),
#                        n_cells_per_crop = (3, 25),
#                        **flow_args
#                        )
    val_dir = data_args['val_dir']
    gen = CoordFlow(val_dir, 
                scale_int = (0, 255.),
                valid_labels = [1],
                is_preloaded = True,
                )
    #%%
    metrics = np.zeros(3)
    N = len(gen.data_indexes)
    inds2check = list(range(N))
    #inds2check = [1, 3, 10, 20]
    for ind in tqdm.tqdm(inds2check):
        image, target = gen.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        
        #image /= image.max()
        
        image = image.to(device)
        #target = {k: v.to(device) for k, v in target.items()}
        
        with torch.no_grad():
            #predictions, (xhat, belive_maps) = model(image[None])
            #predictions, xhat = model(image[None])
            outs = model(image[None], [target])
        
        losses, predictions, (xhat, belive_maps) = outs[:3]
        if len(outs) == 4:
            x_n2n = outs[-1][0,0].cpu().numpy()
            
                
        pred = {k: v.cpu() for k, v in predictions[0].items()}
    
        pred_coords = pred['coordinates'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
            
        target_coords = target['coordinates'].detach().cpu().numpy()
        target_labels = target['labels'].detach().cpu().numpy()
        
        
        TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, target_coords, max_dist = eval_dist, assigment = 'greedy')
        metrics += TP, FP, FN
        
        has_cell_prob =  belive_maps[0, 0].cpu().detach()
        prob_maps = xhat[0, 0].cpu().detach()
        img = image[0].cpu().numpy()
        
        
        
        #figsize = (160, 40)
        #figsize = (80, 20)
        figsize = (20, 10)
        
        if not is_n2n:
            fig, axs = plt.subplots(1, 4, figsize = figsize, sharex = True, sharey = True)
            
        else:
            fig, axs = plt.subplots(1, 5, figsize = figsize, sharex = True, sharey = True)
            axs[-1].imshow(x_n2n, cmap = 'gray')
        
        
        #fig, axs = plt.subplots(1, 4, sharex = True, sharey = True)
        axs[0].imshow(img, cmap = 'gray')
        axs[1].imshow(img, cmap = 'gray')
        axs[2].imshow(prob_maps, vmax = prob_maps.max()*0.5)
        axs[3].imshow(has_cell_prob, vmax = 1.)
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
        
        #fig.savefig('example_andrewjanowczyk.pdf', bbox_inches = 'tight')
        
        #fig.savefig('example_TMA_with_GT.pdf', bbox_inches = 'tight')
        #plt.axis([600, 1000, 600, 1000])
        #fig.savefig('example_TMA_zoomed.pdf', bbox_inches = 'tight')
        
        #%%
    print(model_path)
    TP, FP, FN = metrics
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    score_str = f'P : {P:.3}, R : {R:.3}, F1 : {F1:.3}'
    
    print(score_str)
    