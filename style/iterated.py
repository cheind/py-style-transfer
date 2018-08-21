
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

from style.losses import ContentLoss, GramStyleLoss

class IteratedStyleTransfer:
    def __init__(self, backbone):
        self.backbone = backbone

    def _create_image_tensors(self, p, a, x):        
                
        if p is None:
            p = x if x is not None else a

        p = image.to_np(p)
        a = image.to_np(a)

        if x is None:
            x = image.new_random_white(p.shape, mean=p)
        else:           
            x = image.to_np(x)

        p = image.to_torch(p).to(self.backbone.dev)
        a = image.to_torch(a).to(self.backbone.dev)
        x = image.to_torch(x).to(self.backbone.dev).requires_grad_()    

        return p, a, x    


    def generate(self, 
            content=None,
            style=None,
            seed=None,
            content_loss=None,
            style_loss=None,
            lambda_tv=5e-5,            
            lr=1e-2,
            niter=200,
            yield_every=0,
            disable_progress=False,
            plugins=None):

        plugins = plugins or []
        p, a, x = self._create_image_tensors(content, style, seed)

        opt = optim.Adam([x], lr=lr)
        scheduler = sched.ReduceLROnPlateau(opt, 'min', threshold=1e-3, patience=20, cooldown=50, min_lr=1e-4)

        content_loss = content_loss or ContentLoss(8)
        style_loss = style_loss or GramStyleLoss([6,8,10]) # note, not using conv layer indices by default.

        net = self.backbone.trimmed_net(max(content_loss.layer_ids + style_loss.layer_ids))
        
        with content_loss.create_loss(net) as cl, style_loss.create_loss(net) as sl:
        
            with torch.no_grad():
                net(p); cl.init()
                net(a); sl.init()

            [plugin.prepare(p, a, x, niter=niter) for plugin in plugins]

            losses = None
            with tqdm(total=niter, disable=disable_progress) as t: 
                for idx in range(niter):                  
                   
                    opt.zero_grad()                   

                    net(x)
                    closs = cl() * content_loss.lambda_loss
                    sloss = sl() * style_loss.lambda_loss
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

    def generate_multiscale(self, nlevels=3, **iterate_kwargs):

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
                    content=scale_by(p, f[i]),
                    style=scale_by(a, f[i]),                    
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