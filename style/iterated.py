
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

class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

        self.mean = torch.nn.Parameter(image.vgg_mean.view(1,3,1,1))
        self.std = torch.nn.Parameter(image.vgg_std.view(1,3,1,1))

    def forward(self, x):
        return (x - self.mean) / self.std

class IteratedStyleTransfer:
    def __init__(self, dev=None, avgpool=True):
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vgg = torchvision.models.vgg19(pretrained=True).features[:-1]
        if avgpool:
            layers = [nn.AvgPool2d(2) if isinstance(n, nn.MaxPool2d) else n for n in vgg.children()]            
            vgg = nn.Sequential(*layers)
        
        for param in vgg.parameters():
            param.requires_grad = False

        conv_ids = [idx for idx, m in enumerate(vgg.children()) if isinstance(m, nn.Conv2d)]
        self.conv_ids = np.array(conv_ids)
        
        self.vgg = vgg.eval()
        self.dev = dev 


    def _create_image_tensors(self, p, a, x):        
                
        if p is None:
            p = a

        p = image.to_np(p)
        a = image.to_np(a)

        if x is None:
            x = p.mean((0,1), keepdims=True)
            x = x + np.random.randn(*p.shape).astype(np.float32)*1e-2
        else:           
            x = image.to_np(x)

        p = image.to_torch(p).to(self.dev)
        a = image.to_torch(a).to(self.dev)
        x = image.to_torch(x).to(self.dev).requires_grad_()    

        return p, a, x    

    def _sparse_layer_weights(self, layer_weights, normalize=True):
        layer_weights = np.asarray(layer_weights)
        if normalize:
            layer_weights = layer_weights / layer_weights.sum() 

        layer_ids = np.where(layer_weights != 0)[0]
        layer_weights = layer_weights[layer_ids]

        return layer_ids, layer_weights

    def _create_network(self, content_layer_id, style_layer_ids):
        last_layer = max(content_layer_id, style_layer_ids[-1]) + 1
        
        net = nn.Sequential(
            Normalize(),
            self.vgg[:last_layer]
        ).to(self.dev)

        return net


    def iterate(self, 
            content=None,
            style=None,
            seed=None,
            content_id=8,
            style_weights=None,
            lambda_content=1e-3,
            lambda_style=1e4,
            lambda_tv=5e-5,            
            lr=1e-2,
            niter=200,
            yield_every=50,
            disable_progress=False,
            plugins=None):

        assert len(style_weights) == len(self.conv_ids), 'Need exactly one weight per Conv layer'

        plugins = plugins or []
        
        p, a, x = self._create_image_tensors(content, style, seed)
        style_ids, style_weights = self._sparse_layer_weights(style_weights)
        net = self._create_network(content_id, style_ids)
        
        opt = optim.Adam([x], lr=lr)
        scheduler = sched.ReduceLROnPlateau(opt, 'min', threshold=1e-3, patience=20, cooldown=50, min_lr=1e-4)
        
        with ContentLoss(net[1], content_id) as cl, StyleLoss(net[1], style_ids, style_weights) as sl:
            with torch.no_grad():
                net(p); cl.init()
                net(a); sl.init()

            [plugin.prepare(p,a,x,niter=niter) for plugin in plugins]

            losses = None
            with tqdm(total=niter, disable=disable_progress) as t: 
                for idx in range(niter):                  
                   
                    opt.zero_grad()                   

                    net(x)
                    closs = cl() * lambda_content
                    sloss = sl() * lambda_style
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

                    if idx % yield_every == 0:
                        yield image.to_image(x), losses

        yield image.to_image(x), losses

    def run(self, **iterate_kwargs):
        g = self.iterate(**iterate_kwargs)
        for x in g:
            pass
        return x

    def iterate_multiscale(self, nlevels=3, **iterate_kwargs):

        p = iterate_kwargs.pop('content', None)
        a = iterate_kwargs.pop('style', None)
        _ = iterate_kwargs.pop('seed', None)
        disable = iterate_kwargs.pop('disable_progress', True)

        f = image.pyramid_scale_factors(nlevels)

        x = p.scale_by(f[0])
        with tqdm(total=nlevels) as t: 
            for i in range(nlevels):
                x, losses = self.run(
                    content=p.scale_by(f[i]),
                    style=a.scale_by(f[i]),
                    seed=x,
                    disable_progress=disable,
                    **iterate_kwargs)
                
                if i < nlevels - 1:
                    x = x.up()

                t.set_postfix(loss=np.array_str(losses, precision=3))
                t.update()

                yield x, losses

    def run_multiscale(self, nlevels=3, **iterate_kwargs):
        g = self.iterate_multiscale(nlevels, **iterate_kwargs)
        for x in g:
            pass
        return x

    def _max_lr(self, opt):
        return max([g['lr'] for g in opt.param_groups])    

class ContentLoss:
    
    def __init__(self, net, layer_id):
        self.layer_id = layer_id
        self.hook = net[layer_id].register_forward_hook(self.hookfn)        
        
    def init(self):
        # assumes net(p) called
        self.ref = self.act.data.clone()
        
    def hookfn(self, n, inp, outp):
        self.act = outp
        
    def remove(self):
        self.hook.remove()

    def __call__(self):
        # assumes net(x) called
        return F.mse_loss(self.act, self.ref)
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()

class StyleLoss:
    
    def __init__(self, net, style_layer_ids, style_layer_weights):
        self.layers = [net[l] for l in style_layer_ids]
        self.hooks = [l.register_forward_hook(self.hookfn) for l in self.layers]
        self.prehook = net.register_forward_pre_hook(self.prehookfn)
        self.w = style_layer_weights.tolist()
        self.act = []
        
    def hookfn(self, n, inp, outp):
        self.act.append(outp)
        
    def prehookfn(self, n, inp):
        self.act = []

    def init(self):
        # assumes net(a) called        
        self.A = [self.gram(x).data.clone() for x in self.act]

    def remove(self):
        self.prehook.remove()
        [h.remove() for h in self.hooks]
        
    def __call__(self):
        G = [self.gram(x) for x in self.act]
        E = torch.stack([w * F.mse_loss(g, a).view(-1) for g,a,w in zip(G, self.A, self.w)])       
        return E.sum()
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()
        
    def gram(self, x):
        c, n = x.shape[1], x.shape[2]*x.shape[3]
        f = x.view(c, n)
        return torch.mm(f, f.t()) / (c*n)
