
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

from contextlib import ExitStack

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
            p = x if x is not None else a

        p = image.to_np(p)
        a = image.to_np(a)

        if x is None:
            x = image.new_random_white(p.shape, mean=p)
        else:           
            x = image.to_np(x)

        p = image.to_torch(p).to(self.dev)
        a = image.to_torch(a).to(self.dev)
        x = image.to_torch(x).to(self.dev).requires_grad_()    

        return p, a, x    

    def _sparse_layer_weights(self, layer_weights):
        layer_weights = np.asarray(layer_weights)
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
    
    def style_weights(self, indices_or_dict=None):
        iod = indices_or_dict

        sw = np.zeros(len(self.conv_ids), dtype=np.float32)
        if iod is None: # default
            sw[[6,8,10]] = 1 
        elif isinstance(iod, dict):
            for k,v in iod.items():
                sw[k] = v      
        else:
            for v in iod:
                sw[v] = 1
        
        return sw / sw.sum()


    def generate(self, 
            content=None,
            style=None,
            seed=None,
            content_index=8,
            style_weights=None,
            lambda_content=1e-3,
            lambda_style=1e4,
            lambda_tv=5e-5,            
            lr=1e-2,
            niter=200,
            yield_every=0,
            disable_progress=False,
            plugins=None,
            semantic_style=None,
            semantic_content=None,
            lambda_semantic=1e1):

        plugins = plugins or []

        if style_weights is None:
            style_weights = self.style_weights()

        assert len(style_weights) == len(self.conv_ids), 'Need exactly one weight per Conv layer'
        assert content_index < len(self.conv_ids), 'Convolutional layer not available'

        content_id = self.conv_ids[content_index]

        p, a, x = self._create_image_tensors(content, style, seed)

        style_ids, style_weights = self._sparse_layer_weights(style_weights)
        net = self._create_network(content_id, style_ids)
        
        opt = optim.Adam([x], lr=lr)
        scheduler = sched.ReduceLROnPlateau(opt, 'min', threshold=1e-3, patience=20, cooldown=50, min_lr=1e-4)

        with ExitStack() as stack:
            cl = stack.enter_context(ContentLoss(net[1], content_id))
            if semantic_style is not None:
                assert semantic_content is not None
                sl = stack.enter_context(SemanticStyleLoss(net[1], style_ids, style_weights, semantic_style, semantic_content, lambda_semantic))
            else:
                #sl = stack.enter_context(StyleLoss(net[1], style_ids, style_weights))
                sl = stack.enter_context(PatchStyleLoss(net[1], style_ids, style_weights, k=3, s=3))

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

                    if yield_every > 0 and idx % yield_every == 0:
                        yield image.to_image(x)

        yield image.to_image(x)

    def generate_multiscale(self, nlevels=3, **iterate_kwargs):

        p = iterate_kwargs.pop('content', None)
        a = iterate_kwargs.pop('style', None)
        x = iterate_kwargs.pop('seed', None)
        p_sem = iterate_kwargs.pop('semantic_content', None)
        a_sem = iterate_kwargs.pop('semantic_style', None)

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
                    semantic_style=scale_by(a_sem, f[i]),
                    semantic_content=scale_by(p_sem, f[i]),
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

class PatchStyleLoss(StyleLoss):
    def __init__(self, net, style_layer_ids, style_layer_weights, k=3, s=1):
        super(PatchStyleLoss, self).__init__(net, style_layer_ids, style_layer_weights)
        self.k = k
        self.s = s
        
    def init(self):
        # Extract patches and represent them as kernels of convolutions.
        self.patches = [self._extract_patches(act, self.k, self.s) for act in self.act]
        #self.patches = [self._normalize_patch(p) for p in self.patches]

    def __call__(self):
        
        e = []
        for idx, (pref, act) in enumerate(zip(self.patches, self.act)):
            r = F.conv2d(act, pref, padding=1)
            idx = torch.argmax(r, 1)
            print(act.shape, idx.shape, pref.shape)
            r = torch.index_select(r, 1, idx.squeeze())
            print(r.shape)

            break

        return torch.stack([0]).sum()

    def _normalize_patch(self, x):
        n = torch.norm(x, p=2, dim=1).detach()
        return x.div(n.expand_as(x))

    def _extract_patches(self, x, k, s):
        b, c, h, w = x.shape

        # convolutional arithmetic
        # https://arxiv.org/pdf/1603.07285.pdf

        oh = int((h - k) / s) + 1
        ow = int((w - k) / s) + 1

        oc = oh*ow
        weights = x.new_empty((oc, c, k, k))

        for i in range(oh):
            for j in range(ow):
                weights[i*ow + j] = x[..., i*s:i*s+k, j*s:j*s+k]
        
        return weights

    def _pairwise_distances(self, x, y):
        # Expanded squared norm computation between all pairs of x and y
        x_norm = (x**2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)    
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)


class SemanticStyleLoss(StyleLoss):

    def __init__(self, net, style_layer_ids, style_layer_weights, semantic_style, semantic_content, lambda_semantic=1e1):
        super(SemanticStyleLoss, self).__init__(net, style_layer_ids, style_layer_weights)
        self.semantic_style = image.to_torch(semantic_style)
        self.semantic_content = image.to_torch(semantic_content)
        self.lambda_semantic = lambda_semantic

    def init(self):
        sem = self.act[0].new_tensor(self.semantic_style) * self.lambda_semantic
        self.semantic_stack_style = [F.adaptive_max_pool2d(sem, x.shape[-2:]) for x in self.act]                
        self.A = [self.gram(torch.cat((x,s),1)).data.clone() for x,s in zip(self.act, self.semantic_stack_style)]

    def __call__(self):
        sem = self.act[0].new_tensor(self.semantic_content) * self.lambda_semantic
        sem_stack = [F.adaptive_max_pool2d(sem, x.shape[-2:]) for x in self.act]

        G = [self.gram(torch.cat((x,s),1)) for x,s in zip(self.act, sem_stack)]
        E = torch.stack([w * F.mse_loss(g, a).view(-1) for g,a,w in zip(G, self.A, self.w)])
        return E.sum()
