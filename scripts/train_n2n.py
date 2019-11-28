 i#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from train import data_args, flow_args, load_data, LOG_DIR_DFLT
from fluorescent_counts.flow import FluoMergedFlow

from pathlib import Path 

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
import datetime
import tqdm
import numpy as np

from cell_localization.models import get_mapping_network, model_types
from cell_localization.utils import get_device, save_checkpoint, get_scheduler, get_optimizer

def get_criterion(loss_type):
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'l1smooth':
        criterion = nn.SmoothL1Loss()
    elif loss_type == 'l2':
        criterion = nn.MSELoss()
    else:
        raise ValueError(loss_type)
    return criterion

def trainer_n2n(save_prefix,
        model,
        device,
        train_flow,
        criterion,
        optimizer,
        log_dir,
        lr_scheduler = None,
        batch_size = 16,
        num_workers = 1,
        n_epochs = 2000,
        save_frequency = 200
        ):
    model = model.to(device)
    data_loader = DataLoader(train_flow, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        if lr_scheduler is not None:
            lr_scheduler.step()
        #train
        model.train()
        pbar = tqdm.tqdm(data_loader, desc = f'{save_prefix} Train')        
        train_avg_loss = 0
        for X, target in pbar:
            assert not np.isnan(X).any()
            assert not np.isnan(target).any()
            
            X = X.to(device)
            target = target.to(device)
            pred = model(X)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
            
            train_avg_loss += loss.item()
            
        train_avg_loss /= len(data_loader)
        logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
        
        
        avg_loss = train_avg_loss
        
        desc = 'epoch {} , loss={}'.format(epoch, avg_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)

def train_n2n(
        data_type = 'BBBC038-crops',
        flow_type = None,
        model_name =  'unet-n2n',
        loss_type = 'l1smooth',
        output_type = 'noise2noise',
        cuda_id = 0,
        src_file = None,
        log_dir = None,
        batch_size = 256,
        n_epochs = 500,
        save_frequency = 100,
        num_workers = 2,
        
        optimizer_name = 'adam',
        lr_scheduler_name = '',
        lr = 1e-5,
        weight_decay = 0.0,
        momentum = 0.9,
        
        roi_size = 64,
        
        is_preloaded = True,
        
        train_samples_per_epoch = 40960,
        model_path_init = None
        ):
    
    
    #data_args = data_types[data_type]
    dfl_src_file = data_args['src_file']
    n_ch_in = data_args['n_ch_in']
    n_ch_out = data_args['n_ch_in']
    
    if log_dir is None:
        if 'log_prefix' in data_args:
            log_dir = LOG_DIR_DFLT / output_type / data_args['log_prefix'] / data_type
        else:
            log_dir = LOG_DIR_DFLT / output_type /  data_type
    
    if src_file is None:
        src_file = dfl_src_file
    src_file = Path(src_file)
    cell_crops, bad_crops, backgrounds = load_data(src_file)
    
    train_flow = FluoMergedFlow(cell_crops = cell_crops,
                         bad_crops = bad_crops,
                         backgrounds = backgrounds,
                         
                         output_type = output_type,
                        img_size = (roi_size, roi_size),
                        epoch_size = train_samples_per_epoch,
                        **flow_args
                        )  
    model = get_mapping_network(n_ch_in, n_ch_out, **model_types[model_name])
    
    criterion = get_criterion(loss_type)
    
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
    
    
    save_prefix = f'{data_type}+F{flow_type}+{output_type}+roi{roi_size}_{model_name}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
    trainer_n2n(save_prefix,
        model,
        device,
        train_flow,
        criterion,
        optimizer,
        log_dir = log_dir,
        lr_scheduler = lr_scheduler,
        batch_size = batch_size,
        num_workers = num_workers,
        n_epochs = n_epochs,
        save_frequency = save_frequency
        )
        
if __name__ == '__main__':
    import fire
    fire.Fire(train_n2n)