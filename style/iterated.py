# py-style-transfer
# Copyright 2018 Christoph Heindl.
# Licensed under MIT License
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import numpy as np
from tqdm import tqdm

import style.image as image
import style.priors as priors

from style.losses import Content, GramStyle

class IteratedStyleTransfer:
    '''Iteratively generates an image by minimizing content and style losses.'''

    def __init__(self, backbone):
        '''Initialize from backbone network.'''
        self.backbone = backbone

    def generate(self, 
            style,
            content=None,            
            seed=None,
            lambda_tv=5e-5,            
            lr=1e-2,
            niter=200,
            yield_every=0,
            disable_progress=False,
            plugins=None):
        '''Generate an image by minimizing style and content losses.
        
        Params
        ------
        style : style.losses.LossProvider
            A method that provides style losses. Currently one can choose
            between global GramStyle, PatchStyle and SemanticStyle.
        
        Kwargs
        ------
        content : style.losses.LossProvider
            Provides content loss. If None does not compute a content loss.
        seed : image
            Initial values of image to be generated. If None, initializes with
            a white noise image of size equal to content. If content is not available,
            defaults to white noise of size (256,256,3)
        lambda_tv : scalar
            Strengthness of total variation regularization used to avoid noise artefacts
            in generated image.
        lr : scalar
            Learning rate
        niter : number
            Number of iterations
        yield_every : number
            Yields intermediate results every so often.
        disable_progress : boolean
            Disables progress messages.
        plugins : list, None
            A optional list of optimization plugins to be invoked during various steps of optimization.
        '''


        content = content or Content(layer_id=8)
        plugins = plugins or []

        x = self._get_or_create_seed(content, style, seed)

        opt = optim.Adam([x], lr=lr)
        scheduler = sched.ReduceLROnPlateau(opt, 'min', threshold=1e-3, patience=20, cooldown=50, min_lr=1e-4)

        net = self.backbone.trimmed_net(max(content.layer_ids + style.layer_ids))
        
        with content.create_loss(net, self.backbone.dev) as cl, style.create_loss(net, self.backbone.dev) as sl:
        
            [plugin.prepare(cl, sl, x, niter=niter) for plugin in plugins]

            losses = None
            with tqdm(total=niter, disable=disable_progress) as t: 
                for idx in range(niter):                  
                   
                    opt.zero_grad()                   

                    net(x)
                    closs = cl() * content.lambda_loss
                    sloss = sl() * style.lambda_loss
                    tvloss = priors.tv_prior(x) * lambda_tv
                    loss = closs + sloss + tvloss

                    for plugin in plugins: 
                        loss = plugin.after_loss(x, loss)

                    loss.backward()

                    [plugin.after_backward(x) for plugin in plugins]

                    opt.step()
                    
                    [plugin.after_step(x) for plugin in plugins]
                    
                    losses = np.array((loss.item(), closs.item(), sloss.item(), tvloss.item()))               
                    t.set_postfix(loss=np.array_str(losses, precision=3), lr=self._max_lr(opt))
                    t.update()
                    
                    scheduler.step(loss)

                    # Projected gradient descent
                    x.data.clamp_(0, 1)

                    if yield_every > 0 and idx % yield_every == 0:
                        yield image.to_image(x)

        yield image.to_image(x)

    def _get_or_create_seed(self, content, style, x):        
        if x is None:
            if content.image is None:
                x = image.new_random_white((256,256,3), mean=style.image)
            else:
                x = image.new_random_white(content.image.shape, mean=content.image)

        return image.to_torch(x).to(self.backbone.dev).requires_grad_()    



    def generate_multiscale(self, nlevels=3, **iterate_kwargs):
        '''Generate an image by minimizing style and content losses on multiple image resolutions.'''

        p = iterate_kwargs.pop('content', None)
        a = iterate_kwargs.pop('style', None)
        x = iterate_kwargs.pop('seed', None)

        disable = iterate_kwargs.pop('disable_progress', True)
        yield_every = iterate_kwargs.pop('yield_every', 0)
    
        f = image.pyramid_scale_factors(nlevels)

        def scale_by(x, f):
            if x is not None:
                x = x.scale_by(f)
            return x

        x = scale_by(x, f[0])
        with tqdm(total=nlevels) as t: 
            for i in range(nlevels):
                g = self.generate(
                    content=p.scale_by(f[i]),
                    style=a.scale_by(f[i]),                    
                    seed=x,
                    disable_progress=disable,
                    yield_every=0,
                    **iterate_kwargs)

                x = next(g)

                if i < nlevels - 1:
                    x = x.up()

                t.update()

                if yield_every > 0:
                    yield x # yield intermediate results
        
        if yield_every == 0:
            yield x

    def _max_lr(self, opt):
        return max([g['lr'] for g in opt.param_groups])