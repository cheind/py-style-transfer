
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import numpy as np

from style.transforms import mean, std, normalize, denormalize

class IteratedStyleTransfer:
    def __init__(self, dev=None, avgpool=True):
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        net = torchvision.models.vgg19(pretrained=True).features[:-1]
        if avgpool:
            layers = [nn.AvgPool2d(2) if isinstance(n, nn.MaxPool2d) else n for n in net.children()]
            net = nn.Sequential(*layers)
        
        for param in net.parameters():
            net.requires_grad = False


        conv_ids = [idx for idx, m in enumerate(net.children()) if isinstance(m, nn.Conv2d)]
        self.conv_ids = np.array(conv_ids)
        
        self.net = net.eval().to(dev)
        self.dev = dev

    def iterate(self, p, a, cid, sids, niter=200, lr=1e-2, wc=1, ws=1e3, x=None):
        from tqdm import tqdm
    
        p = normalize(p).to(self.dev).unsqueeze(0)
        a = normalize(a).to(self.dev).unsqueeze(0)    

        # Larger noise leads to more diverse images, but convergence is slower.
        if x is None:
            x = torch.tensor(torch.randn_like(p)*5e-2, requires_grad=True).to(self.dev)
        else:
            x = normalize(x).to(self.dev).unsqueeze(0).requires_grad_()
        
        opt = optim.Adam([x], lr=lr)
        scheduler = sched.ReduceLROnPlateau(opt, 'min', threshold=1e-3, patience=20, cooldown=50, min_lr=1e-3)

        last_layer = max(cid, sids[-1]) + 1
        net = self.net[:last_layer]
        
        with ContentLoss(net, cid) as cl, StyleLoss(net, sids) as sl:
            with torch.no_grad():
                net(p); cl.init()
                net(a); sl.init()

            xmin = -mean / std
            xmax = (1-mean) / std

            with tqdm(total=niter) as t: 
                for idx in range(niter):                  
                   
                    opt.zero_grad()
                    x.data[:,0].clamp_(xmin[0], xmax[0])
                    x.data[:,1].clamp_(xmin[1], xmax[1])
                    x.data[:,2].clamp_(xmin[2], xmax[2])

                    net(x)
                    closs = cl() * wc
                    sloss = sl() * ws
                    loss = closs + sloss
                    loss.backward()

                    opt.step()
                    
                    losses = np.array((closs.item(), sloss.item(), loss.item()))               
                    t.set_postfix(loss=np.array_str(losses, precision=3), lr=self._max_lr(opt))
                    t.update()
                    
                    scheduler.step(loss)
                    
                    if idx % 50 == 0:
                        yield x
        yield x

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
    
    def __init__(self, net, layer_ids):
        self.layer_ids = layer_ids
        self.layers = [net[l] for l in layer_ids]
        self.hooks = [l.register_forward_hook(self.hookfn) for l in self.layers]
        self.prehook = net.register_forward_pre_hook(self.prehookfn)
        self.w = 1 / len(layer_ids)
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
        E = torch.stack([F.mse_loss(g, a) for g,a in zip(G, self.A)])        
        return E.mean()
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.remove()
        
    def gram(self, x):
        c, n = x.shape[1], x.shape[2]*x.shape[3]
        f = x.view(c, n)
        return torch.mm(f, f.t()) / (c*n)