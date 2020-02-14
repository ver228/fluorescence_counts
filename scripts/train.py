    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

#from config_opts import flow_types, data_types
from fluorescence_counts.unet_n2n import ModelWithN2N, MockModel
from fluorescence_counts.flow import FluoMergedFlow, load_data

from cell_localization.utils import get_device, get_scheduler, get_optimizer
from cell_localization.trainer import train_locmax, train_segmentation
from cell_localization.flow import CoordFlow, FlowCellSegmentation
from cell_localization.models import get_model

from torch import nn
import datetime
import torch




data_args = dict(
        #src_file = Path.home() / 'workspace/localization/data/crops4synthetic/BBBC038_Kaggle_2018_Data_Science_Bowl.p',
        src_file = Path.home() / 'workspace/localization/data/crops4synthetic/BBBC038_Kaggle_2018_Data_Science_Bowl_v2.p',
        val_dir = Path.home() / 'workspace/localization/data/BBBC038_Kaggle_2018_Data_Science_Bowl//fluorescence/validation/',
    
        n_ch_in = 1
        )

flow_args = dict(
        int_scale = (0, 255),
                         
         n_cells_per_crop = (0, 7),
         crop_intensity_range = (0.2, 1.),
         fg_quantile_range = (0, 10),
         bg_quantile_range = (25, 75),
         
         frac_crop_valid = 0.1,
         zoom_range = (0.9, 1.1),
         rotate_range = (0, 90),
         max_overlap = 0.4,
         
         null_value = 0.,
         merge_by_prod = False
        )
        

def train(
        data_type = 'BBBC038-crops',
        flow_type = None,
        model_name =  'clf+unet-simple+n2n', #'clf+unet-simple',#
        loss_type = 'maxlikelihood',#'l1smooth-G1.5', #
        coord_type = 'centroids+noise2noise',#'point-inside-contour', #
        cuda_id = 0,
        log_dir = None,
        batch_size = 64, #256
        n_epochs = 2000,
        save_frequency = 500,
        num_workers = 2,
        src_file = None,
        
        optimizer_name = 'adam',
        lr_scheduler_name = '',
        lr = 1e-5,
        weight_decay = 0.0,
        momentum = 0.9,
        
        roi_size = 128, #64
        
        is_preloaded = False,
        
        hard_mining_freq = None,
        model_path_init = None,
        train_samples_per_epoch = 10240,#40960,
        
        num_folds = None,
        val_fold_id = None,
        
        val_dist = 5,
        crops_same_image = False,
        ignore_bad_crops = False
        ):
    
    
    #data_args = data_types[data_type]
    dfl_src_file = data_args['src_file']
    
    n_ch_in = data_args['n_ch_in']
    
    val_dir = data_args['val_dir']
    #if flow_type is None:
    #    flow_type = data_args['dflt_flow_type']
    #flow_args = flow_types[flow_type]
    
    
    if model_name.startswith('seg+'):
        log_dir_dflt = Path.home() / 'workspace/localization/results/segmentation'
        n_ch_out = 3
        trainer = train_segmentation
        val_flow_func = FlowCellSegmentation
        n2n_lambda_criterion = 10
    else:
        log_dir_dflt = Path.home() / 'workspace/localization/results/locmax_detection'
        n_ch_out = data_args['n_ch_in']
        trainer = train_locmax
        val_flow_func = CoordFlow
        n2n_lambda_criterion = 100 if  'maxlikelihood' in loss_type else 1
    
    
    if log_dir is None:
        if 'log_prefix' in data_args:
            log_dir = log_dir_dflt / data_args['log_prefix'] / data_type
        else:
            log_dir = log_dir_dflt / data_type
    
    if src_file is None:
        src_file = dfl_src_file
    src_file = Path(src_file)
    cell_crops, bad_crops, backgrounds = load_data(src_file, 
                                                   min_size = 2, 
                                                   crops_same_image = crops_same_image,
                                                   ignore_bad_crops = ignore_bad_crops
                                                   )
    
    train_flow = FluoMergedFlow(cell_crops = cell_crops,
                         bad_crops = bad_crops,
                         backgrounds = backgrounds,
                         
                         output_type = coord_type,
                        img_size = (roi_size, roi_size),
                        max_crop_size = roi_size,
                        epoch_size = train_samples_per_epoch,
                        
                        **flow_args
                        )  
    
    val_flow = val_flow_func(val_dir, 
                scale_int = (0, 255.),
                valid_labels = [1],
                is_preloaded = True,
                )
    
    if model_name == 'mock+n2n':
        model = MockModel(n_ch_in, n_ch_out)
        model = ModelWithN2N(n_ch_in, 
                             model, 
                             freeze_n2n = False,
                             criterion_n2n = nn.SmoothL1Loss(),
                             n2n_lambda_criterion = n2n_lambda_criterion
                             )
        
    else:
        model_name_r, is_n2n, n2n_type = model_name.partition('+n2n')
        model = get_model(model_name_r, n_ch_in, n_ch_out, loss_type)
        
        if is_n2n:
            n2n_types = n2n_type.split('+')
            
            detach_n2n = 'detach' in n2n_types
            
            if 'frozen' in n2n_types:
                model = ModelWithN2N(n_ch_in, model, n2n_freeze = True)
                n2n_model_path = Path.home() / 'workspace/localization/results/locmax_detection/noise2noise/BBBC038-crops/BBBC038-crops+FNone+noise2noise+roi64_unet-n2n_l1smooth_20191114_121612_adam_lr1e-05_wd0.0_batch256/checkpoint-99.pth.tar'
                state_n2n = torch.load(n2n_model_path, map_location = 'cpu')
                model.n2n.load_state_dict(state_n2n['state_dict'])
            else:
                n2n_criterion = nn.SmoothL1Loss() if 'noise2noise' in coord_type else None
                    
                model = ModelWithN2N(n_ch_in, 
                                     model, 
                                     n2n_freeze = False,
                                     n2n_criterion = n2n_criterion,
                                     n2n_lambda_criterion = n2n_lambda_criterion,
                                     detach_n2n = detach_n2n
                                     )
    
    

    if model_path_init is not None:
        model_name += '-pretrained'
        state = torch.load(model_path_init, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
    
    device = get_device(cuda_id)
    
    optimizer = get_optimizer(optimizer_name, 
                              model, 
                              lr = lr, 
                              momentum = momentum, 
                              weight_decay = weight_decay)
    
    lr_scheduler = get_scheduler(lr_scheduler_name, optimizer)
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    lr_scheduler_name = '+' + lr_scheduler_name if lr_scheduler_name else ''
    
    if crops_same_image:
        data_type += '+sameimages'
        
    if ignore_bad_crops:
        data_type += '+nobadcrops'
    
    save_prefix = f'{data_type}+F{flow_type}+{coord_type}+roi{roi_size}_{model_name}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
    
    trainer(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        optimizer,
        lr_scheduler = lr_scheduler,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        hard_mining_freq = hard_mining_freq,
        n_epochs = n_epochs,
        save_frequency = save_frequency,
        val_dist = val_dist
        )

if __name__ == '__main__':
    import fire
    fire.Fire(train)
    