#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

"""
import math
import torch
import torch.nn.functional as F

from torch import nn
from cell_localization.models import get_mapping_network

class MockModel(nn.Module):
    def __init__(self, n_in, n_out, **args):
        super().__init__()
        self.n_classes = n_out
        self.input_parameters = {}
    
    def forward(self, x, targets = None):
        losses = {}
        if self.training:
            return losses
        else:
            preds = {
                    'coordinates' : torch.zeros((0, 2)),
                    'labels' : torch.zeros((0))
                    }
            
            return losses, preds
        


class ModelWithN2N(nn.Module):
    def __init__(self, 
                 n_channels, 
                 main_model, 
                 n2n_freeze = False,
                 n2n_criterion = None,
                 n2n_lambda_criterion = 100,
                 n2n_return = False,
                 detach_n2n = False
                 ):
        super().__init__()
        
        self.main_model = main_model
        self.n_classes = main_model.n_classes
        self.input_parameters = main_model.input_parameters
        
        self.n2n_return = n2n_return
        self.n2n_freeze = n2n_freeze
        self.detach_n2n = detach_n2n
        
        self.n2n = get_mapping_network(n_channels, n_channels, model_type = 'unet-n2n')
        
        if self.n2n_freeze:
            for param in self.n2n.parameters():
                param.requires_grad = False
        else:
            self.n2n_criterion = n2n_criterion
            self.n2n_lambda_criterion = n2n_lambda_criterion
            
            
    def forward(self, x, targets = None):
        x_n2n = self.n2n(x)
        xin = x_n2n.detach() if self.detach_n2n else x_n2n
        outs = self.main_model(xin, targets)
        
        if not self.n2n_freeze  and self.n2n_criterion is not None and self.training and (targets is not None):
            x_target = torch.stack([t['n2n_target'] for t in targets])
            n2n_loss = self.n2n_lambda_criterion*self.n2n_criterion(x_n2n, x_target)
            
            if isinstance(outs, dict):
                #in this case the losses is expected to be in the returned element
                outs['n2n_loss'] = n2n_loss
            else:
                outs[0]['n2n_loss'] = n2n_loss
        
        if self.n2n_return:
            if isinstance(outs, (list, tuple)):
                outs = (*outs, x_n2n)
            else:
                outs = outs, x_n2n
        
        return outs
        
    
def weights_init_xavier(m):
    '''
    Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.startswith('Linear'):
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.startswith('BatchNorm2d'):
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def _crop(x, x_to_crop):
    c = (x_to_crop.size()[2] - x.size()[2])/2
    c1, c2 =  math.ceil(c), math.floor(c)
    
    c = (x_to_crop.size()[3] - x.size()[3])/2
    c3, c4 =  math.ceil(c), math.floor(c)
    
    cropped = F.pad(x_to_crop, (-c3, -c4, -c1, -c2)) #negative padding is the same as cropping
    
    return cropped
def _conv3x3(n_in, n_out):
    return [nn.Conv2d(n_in, n_out, 3, padding=1),
    nn.LeakyReLU(negative_slope=0.1, inplace=True)]

class Down(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        _layers = _conv3x3(n_in, n_out) + [nn.MaxPool2d(2)]
        self.conv_pooled = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv_pooled(x)
        return x


class Up(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += _conv3x3(n_in, n_out)
        self.conv = nn.Sequential(*_layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = _crop(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNetN2N(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1):
        super().__init__()
        self.conv0 = nn.Sequential(*_conv3x3(n_channels, 48))
        
        self.down1 = Down(48, 48)
        self.down2 = Down(48, 48)
        self.down3 = Down(48, 48)
        self.down4 = Down(48, 48)
        self.down5 = Down(48, 48)
        
        self.conv6 = nn.Sequential(*_conv3x3(48, 48))
        
        self.up5 = Up([96, 96, 96])
        self.up4 = Up([144, 96, 96])
        self.up3 = Up([144, 96, 96])
        self.up2 = Up([144, 96, 96])
        self.up1 = Up([96 + n_channels, 64, 32])
        
        self.conv_out = nn.Sequential(nn.Conv2d(32, n_classes, 3, padding=1))
        
        for m in self.modules():
            weights_init_xavier(m)
    
    def _unet(self, x_input):    
        x0 = self.conv0(x_input)
        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        
        x6 = self.conv6(x5)
        
        x = self.up5(x6, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x_input)
        
        x = self.conv_out(x)
        return x
    
    def forward(self, x_input):
        # the input shape must be divisible by 32 otherwise it will be cropped due 
        #to the way the upsampling in the network is done. Therefore it is better to path 
        #the image and recrop it to the original size
        nn = 2**5
        ss = [math.ceil(x/nn)*nn - x for x in x_input.shape[2:]]
        pad_ = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
        
        #use pytorch for the padding
        pad_ = [x for d in pad_[::-1] for x in d] 
        pad_inv_ = [-x for x in pad_] 
        
        
        x_input = F.pad(x_input, pad_, 'reflect')
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        
        return x
        
#    def forward(self, x_input):
#        x0 = self.conv0(x_input)
#        
#        x1 = self.down1(x0)
#        x2 = self.down2(x1)
#        x3 = self.down3(x2)
#        x4 = self.down4(x3)
#        x5 = self.down5(x4)
#        
#        x6 = self.conv6(x5)
#        
#        x = self.up5(x6, x4)
#        x = self.up4(x, x3)
#        x = self.up3(x, x2)
#        x = self.up2(x, x1)
#        x = self.up1(x, x_input)
#        
#        x = self.conv_out(x)
#        return x



if __name__ == '__main__':
    
    model = MockModel(1, 1)
    
    model = ModelWithN2N(1, model, n2n_criterion = nn.SmoothL1Loss(), n2n_return = True)
    X = torch.rand((1, 1, 540, 600))
    targets = [{'n2n_target' : torch.rand((1, 540, 600))}]
    
    #%%
    losses, x_out = model(X, targets)
    loss = sum([x for x in losses.values()])
    loss.backward()
    
    print(x_out.size())
    