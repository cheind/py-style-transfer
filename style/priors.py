# py-style-transfer
# Copyright 2018 Christoph Heindl.
# Licensed under MIT License
# ============================================================

import torch
import torch.nn.functional as F

def tv_prior(x):
    '''Computes the total variation loss'''

    xdir = x.new_tensor([[-1, 1]]).view(1,1,1,2).repeat(3,1,1,1)
    ydir = x.new_tensor([[-1],[1]]).view(1,1,2,1).repeat(3,1,1,1)


    fx = F.conv2d(x, xdir, padding=0, groups=3)
    fy = F.conv2d(x, ydir, padding=0, groups=3)

    return torch.cat([
        fx.abs().view(-1), 
        fy.abs().view(-1)]).mean()