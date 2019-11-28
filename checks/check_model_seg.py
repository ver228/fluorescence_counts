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

from cell_localization.evaluation import segmentation2contours

if __name__ == '__main__':
    
    #nms_threshold_rel = 0.2
    cuda_id = 0
    device = get_device(cuda_id)
    results_dir = Path.home() / 'workspace/localization/results/segmentation/BBBC038-crops/'
    
    #bn = 'BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-5_20191126_094934_adam_lr0.000128_wd0.0_batch128'
    #results_dir = Path.home() / 'workspace/localization/results/segmentation/BBBC038/fluorescence/BBBC038-fluorescence/'
    
    bn = 'BBBC038-fluorescence+Flymphocytes+roi96_seg+unet-resnet101_crossentropy+W1-1-10_20191120_213741_adam_lr0.000128_wd0.0_batch128'
    results_dir = Path.home() / 'Desktop/nuclei_datasets/'
    #model_path = results_dir / bn / 'checkpoint-199.pth.tar'
    
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
    
    model = get_model(model_type, 
                      n_ch_in, 
                      n_ch_out, 
                      loss_type,
                      return_belive_maps = True
                      )
    if is_n2n:
        model = ModelWithN2N(n_ch_in, model, n2n_return = True)
    #%%
    state = torch.load(model_path, map_location = 'cpu')
    #%%
    model.load_state_dict(state['state_dict'])
    model.eval()
    model = model.to(device)
    #%%
    fnames = [
            '/Users/avelinojaver/Desktop/nuclei_datasets/BBBC038_Kaggle_2018_Data_Science_Bowl/stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png',
            '/Users/avelinojaver/Desktop/h_0_1373_w_12831_14664.jpg',
            '/Users/avelinojaver/Desktop/h_20595_21968_w_0_1833.jpg',
            '/Users/avelinojaver/OneDrive - Nexus365/heba/spheroids/sample_nanog_mastrixplate2_20x/H - 6(fld 24 wv UV - DAPI z 4).tif',
            ]
    #%%
    for fname in fnames[-1:]:
        
        #ind = 17
        #image, target = val_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = cv2.imread(fname, cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float32)
        #image = image[:512, -512:]
        
        
        if max(image.shape) > 1000:
            image =  image[800:1300, 900:1400]
            image /= image.max()
        #else:
        #image /= 255
        #image /= image.max()
        image = torch.from_numpy(image[None])
        
        image = image.to(device)
        #target = {k: v.to(device) for k, v in target.items()}
        
        with torch.no_grad():
            outs = model(image[None])
        
        predictions, xhat = outs[:2]
        if len(outs) == 3:
            x_n2n = outs[-1][0,0].cpu().numpy()
        
        
        labels = predictions[0].cpu().detach().numpy()
        
        prob_maps = torch.softmax(xhat, dim=1)
        prob_maps = np.rollaxis(prob_maps[0].cpu().detach().numpy(), 0, 3)
        
        img = image[0].cpu().numpy()
        
        
        
        cell_contours = segmentation2contours(labels == 1)
        
        #img_inv = np.round((1-img)*255).astype(np.uint8)[..., None].repeat(3, axis = 2)
        #img_inv[labels==0] = 255
        
        #markers = cv2.watershed(img_inv, seg_mask.copy())
        #markers[labels==0] = 0
        #plt.imshow(markers)
        
        
        
        figsize = (80, 20)
        #figsize = (40, 10)
        #figsize = (5, 20)
        #figsize = None
        
        n_subplots = 4 + (1 if is_n2n else 0)
        fig, axs = plt.subplots(1, n_subplots, figsize =figsize, sharex = True, sharey = True)
        #fig, axs = plt.subplots(1, n_subplots,  sharex = True, sharey = True)
        
        if is_n2n:
            axs[1].imshow(x_n2n, cmap = 'gray')
        
        axs[0].imshow(img, cmap = 'gray')
        
        
        axs[-1].imshow(img, cmap = 'gray')
        for cnt in cell_contours:
            axs[-1].plot(cnt[:, 0], cnt[:, 1], 'r')
        
        
        axs[-3].imshow(prob_maps, vmax = prob_maps.max()*0.5)
        axs[-2].imshow(labels)
        for ax in axs:
            ax.axis('off')
        
        fig.savefig(Path(fname).stem + '_seg.png', bbox_inches = 'tight')
        