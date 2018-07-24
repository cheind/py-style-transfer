
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import numpy as np
from tqdm import tqdm

from style.transforms import Normalize, to_tensor
from style.priors import tv_prior, tv_prior2

class IteratedStyleTransfer:
    def __init__(self, dev=None, avgpool=True):
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vgg = torchvision.models.vgg19(pretrained=True).features[:-1]
        if avgpool:
            layers = [nn.AvgPool2d(2) if isinstance(n, nn.MaxPool2d) else n for n in vgg.children()]
            net = nn.Sequential(*layers)
        
        for param in net.parameters():
            param.requires_grad = False


        conv_ids = [idx for idx, m in enumerate(vgg.children()) if isinstance(m, nn.Conv2d)]
        self.conv_ids = np.array(conv_ids)
        
        self.vgg = vgg.eval()
        self.dev = dev        

    def iterate(self, 
            p, a, content_layer_id, style_layer_weights, 
            x=None,
            weight_content_loss=1e-3,
            weight_style_loss=1e4,
            weight_tv_loss=5e-5,
            niter=200,
            yield_freq=50,
            lr=1e-2):

        assert len(style_layer_weights) == len(self.conv_ids), 'Need exactly one weight per Conv layer'
        
        p = to_tensor(p).to(self.dev).unsqueeze(0)
        a = to_tensor(a).to(self.dev).unsqueeze(0)    

        # Larger noise leads to more diverse images, but convergence is slower.
        if x is None:
            data = p.view(p.shape[0], p.shape[1], -1).mean(-1).view(-1,3,1,1) + torch.randn_like(p)*1e-2
            x = torch.tensor(data, requires_grad=True).to(self.dev)            
        else:
            x = to_tensor(x).to(self.dev).unsqueeze(0).requires_grad_()

        style_layer_weights = np.asarray(style_layer_weights)
        style_layer_weights = style_layer_weights / style_layer_weights.sum() 
        style_layer_ids = np.where(style_layer_weights != 0)[0]
        style_layer_weights = style_layer_weights[style_layer_ids]
        
        last_layer = max(content_layer_id, style_layer_ids[-1]) + 1

        net = nn.Sequential(
            Normalize(),
            self.vgg
        ).to(self.dev)

        opt = optim.Adam([x], lr=lr)
        scheduler = sched.ReduceLROnPlateau(opt, 'min', threshold=1e-3, patience=20, cooldown=50, min_lr=1e-4)
        
        with ContentLoss(net[1], content_layer_id) as cl, StyleLoss(net[1], style_layer_ids, style_layer_weights) as sl:
            with torch.no_grad():
                net(p); cl.init()
                net(a); sl.init()

            with tqdm(total=niter) as t: 
                for idx in range(niter):                  
                   
                    opt.zero_grad()                   

                    net(x)
                    closs = cl() * weight_content_loss
                    sloss = sl() * weight_style_loss
                    tvloss = tv_prior(x) * weight_tv_loss
                    loss = closs + sloss + tvloss
                    loss.backward()

                    opt.step()
                    
                    losses = np.array((loss.item(), closs.item(), sloss.item(), tvloss.item()))               
                    t.set_postfix(loss=np.array_str(losses, precision=3), lr=self._max_lr(opt))
                    t.update()
                    
                    scheduler.step(loss)

                    # Projected gradient descent
                    x.data.clamp_(0, 1)

                    if idx % yield_freq == 0:
                        yield x, losses
        yield x, losses

    def run(self, *args, **kwargs):
        g = self.iterate(*args, **kwargs)
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