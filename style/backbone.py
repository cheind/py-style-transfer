# py-style-transfer
# Copyright 2018 Christoph Heindl.
# Licensed under MIT License
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from collections import Iterable

from style.image import vgg_mean, vgg_std

class Normalize(torch.nn.Module):
    '''Normalize image tensor with training mean/std.'''
    
    def __init__(self):
        super(Normalize, self).__init__()

        self.mean = torch.nn.Parameter(vgg_mean.view(1,3,1,1))
        self.std = torch.nn.Parameter(vgg_std.view(1,3,1,1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Backbone:
    '''Backbone network used to compute feature based losses.'''

    def __init__(self, dev=None, avgpool=True):
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = dev

        vgg = torchvision.models.vgg19(pretrained=True).features[:-1]
        if avgpool:
            layers = [nn.AvgPool2d(2) if isinstance(n, nn.MaxPool2d) else n for n in vgg.children()]            
            vgg = nn.Sequential(*layers)
        
        for param in vgg.parameters():
            param.requires_grad = False

        self.net = nn.Sequential(
            Normalize(),
            *vgg
        ).to(self.dev).eval()

        conv_ids = [idx for idx, m in enumerate(self.net.children()) if isinstance(m, nn.Conv2d)]
        self.conv_ids = np.array(conv_ids)

    def trimmed_net(self, last_layer_index):
        '''Returns the network trimmed to last used layer.'''
        return self.net[:(last_layer_index + 1)]


    def conv_layer_index(self, idx):
        '''Converts one or more convolutional index to network layer index.'''
        if isinstance(idx, Iterable):
            return [self.conv_ids[l] for l in idx]
        else:
            return self.conv_ids[idx]
        
    
    
