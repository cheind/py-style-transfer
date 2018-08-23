# py-style-transfer
# Copyright 2018 Christoph Heindl.
# Licensed under MIT License
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F

from style.image import to_np, to_torch, to_image, NEAREST

class BaseLoss:
    '''Base loss functor. 
    
    Modelled as context manager so that any network hooks are properly 
    freed after optimization. The optimization will iteratively pass
    the current state of image reconstruction through the backbone network.
    Losses should therefore hook different layers of the network and 
    gather network responses. Upon `__call__` the recorded values are supposed
    to be converted into a loss of somesort and returned.
    '''

    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, type, value, tb):
        self.remove()

    def enter(self):
        '''Called upon entering of context.'''
        pass

    def remove(self):
        '''Called upon leaving the context.'''
        pass

    def __call__(self):
        '''Returns the loss.'''
        raise NotImplementedError()

class NoopLoss(BaseLoss):
    ''' A loss that does nothing.'''
    def __init__(self, dev):
        self.z = torch.tensor([0], dtype=torch.float32).to(dev)

    def __call__(self):
        return self.z

class LossProvider(object):
    '''Base class for providing losses.'''

    def __init__(self, layer_ids, lambda_loss):
        self.layer_ids = layer_ids
        self.lambda_loss = lambda_loss

    def create_loss(self, net, dev):
        raise NotImplementedError()

class Content(LossProvider):
    '''Content loss provider.

    Computes the mean squared error of reconstruction activations and 
    corresponding content image activations.

    Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. 
    "A neural algorithm of artistic style."
    arXiv preprint arXiv:1508.06576 (2015).
    '''

    def __init__(self, image=None, layer_id=8, lambda_loss=1e-3):
        '''Create Content loss provider.

        Kwargs
        ------
        image : image
            Style image to be matched.
        layer_id : number
            Layer index to compute feature losses at.
        lambda_loss : scalar
            Lagrange multiplier strengthness for this loss vs. other losses.
        '''

        super(Content, self).__init__([layer_id], lambda_loss)

        if image is not None:
            self.image = to_np(image)
        else:
            self.image = None

    def create_loss(self, net, dev):
        if self.image is not None:
            return Content.Loss(net, dev, self.layer_ids[0], self.image)
        else:
            return NoopLoss(dev)

    def scale_by(self, f):
        image = None
        if self.image is not None:
            image = to_image(self.image).scale_by(f)           

        return Content(image, self.layer_ids[0], self.lambda_loss)


    class Loss(BaseLoss):
        def __init__(self, net, dev, lid, image):
            self.net = net
            self.lid = lid
            self.dev = dev
            self.image = to_torch(image).to(dev)
            
        def enter(self):
            self.hook = self.net[self.lid].register_forward_hook(self.hookfn)
            self.init()

        def hookfn(self, n, inp, outp):
            self.act = outp

        def remove(self):
            self.hook.remove()

        def init(self):
            with torch.no_grad():
                self.net(self.image)
                self.ref = self.act.data.clone()
                
        def __call__(self):
            # assumes net(x) called
            return F.mse_loss(self.act, self.ref)


class GramStyle(LossProvider):
    '''Gram style loss provider.

    Computes the style loss between the reconstruction and style image as described in 
    https://arxiv.org/pdf/1508.06576.pdf
    '''

    def __init__(self, image=None, layer_ids=None, layer_weights=None, lambda_loss=1e4):   
        '''Create GramStyle.

        Kwargs
        ------
        image : image
            Style image to be matched.
        layer_ids : list
            Optional list of raw network layers to compute feature losses at.
            If None defaults to [6,8,10].
        layer_weights : list
            Optional weights for each layer loss. If None uniform weights are  
            applied.
        lambda_loss : scalar
            Lagrange multiplier strengthness for this loss vs. other losses.
        '''

        if layer_ids is None:
            layer_ids = [6,8,10]

        super(GramStyle, self).__init__(layer_ids, lambda_loss)
        self.layer_weights = layer_weights
        self.image = to_np(image)

    def create_loss(self, net, dev):
        return GramStyle.Loss(net, dev, self.layer_ids, self.layer_weights, self.image)

    def scale_by(self, f):
        image = to_image(self.image).scale_by(f) 
        return GramStyle(image, self.layer_ids, self.layer_weights, self.lambda_loss)

    class Loss(BaseLoss):
        def __init__(self, net, dev, lids, w, image):
            self.lids = lids
            self.w = w
            self.net = net
            self.image = to_torch(image).to(dev)

        def enter(self):
            layers = [self.net[l] for l in self.lids]
            self.hooks = [l.register_forward_hook(self.hookfn) for l in layers]
            self.prehook = self.net.register_forward_pre_hook(self.prehookfn)

            if self.w is None:
                self.w = np.ones(len(self.lids), dtype=np.float32) / len(self.lids)
                self.w = self.w.tolist()
            else:
                assert len(self.w) == len(self.lids)
                self.w = np.asarray(self.w).tolist()

            self.act = []
            self.init()

        def hookfn(self, n, inp, outp):
            self.act.append(outp)

        def prehookfn(self, n, inp):
            self.act = []

        def remove(self):
            [h.remove() for h in self.hooks]
            self.prehook.remove()

        def init(self):
            with torch.no_grad():
                self.net(self.image)
                self.A = [self.gram(x).data.clone() for x in self.act]

        def __call__(self):
            G = [self.gram(x) for x in self.act]
            E = torch.stack([w * F.mse_loss(g, a).view(-1) for g,a,w in zip(G, self.A, self.w)])      
            return E.sum()

        def gram(self, x):
            c, n = x.shape[1], x.shape[2]*x.shape[3]
            f = x.view(c, n)
            return torch.mm(f, f.t()) / (c*n)

class PatchStyle(GramStyle):
    '''Patch style loss provider.

    Computes the style loss between the reconstruction and style image as described in 
    https://arxiv.org/pdf/1601.04589.pdf
    '''

    def __init__(self, image=None, layer_ids=None, layer_weights=None, lambda_loss=1e-2, k=3, s=1):   
        '''Create PatchStyle.

        Kwargs
        ------
        image : image
            Style image to be matched.
        layer_ids : list
            Optional list of raw network layers to compute feature losses at.
            If None defaults to [6,8,10].
        layer_weights : list
            Optional weights for each layer loss. If None uniform weights are  
            applied.
        lambda_loss : scalar
            Lagrange multiplier strengthness for this loss vs. other losses.
        k : number
            Kernel size
        s : number
            Stride
        '''     
        super(PatchStyle, self).__init__(image, layer_ids, layer_weights, lambda_loss)
        self.k = k
        self.s = s

    def create_loss(self, net, dev):
        return PatchStyle.Loss(net, dev, self.layer_ids, self.layer_weights, self.image, self.k, self.s)

    def scale_by(self, f):
        image = to_image(self.image).scale_by(f) 
        return PatchStyle(image, self.layer_ids, self.layer_weights, self.lambda_loss, self.k, self.s)

    class Loss(GramStyle.Loss):

        def __init__(self, net, dev, lids, w, image, k, s):
            super(PatchStyle.Loss, self).__init__(net, dev, lids, w, image)
            self.k = k
            self.s = s

        def init(self):
            # Called from enter of GramStyle
            with torch.no_grad():
                self.net(self.image)
                self.style_act = [a.detach().clone() for a in self.act]  

        def __call__(self):
            e = []
            for sa, a, w in zip(self.style_act, self.act, self.w):
                pa, psa = self.nearest(a, sa, kx=self.k, ky=self.k, sy=self.s, sx=1)
                e.append(w * F.mse_loss(pa, psa).view(-1))
            return torch.cat(e).sum()

        def nearest(self, x, y, kx=3, sx=1, ky=3, sy=1):
            '''Returns the nearest neighbor patch in y for every patch in x
            according to normalized cross correlation.

            Params
            ------
            x : 1xCxHxW tensor
            y : 1xCxH'xW' tensor

            Returns
            -------
            px : NxK*K*C tensor
            py : NxK*K*C tensor

            with N being the number of patches in x.    
            '''
            py = F.unfold(y, ky, stride=sy).transpose(1,2).squeeze(0) # nxp
            ny = torch.norm(py, 2, 1)

            px = F.unfold(x, kx, stride=sx).transpose(1,2).squeeze(0) # mxp
            
            # Treat px constant during nearest search, allows for splitting
            # the matrix product. If gradients would be on, we could still split
            # the matrix product, but memory would need to be kept for backward.
            pxd = px.detach()
            
            nx = torch.norm(pxd, 2, 1) # treat norm as constant during opt
            
            nxs = torch.split(nx, 1024, dim=0)
            pxds = torch.split(pxd, 1024, dim=0)
            
            nids = []
            for pxd_part, nx_part in zip(pxds, nxs):
                d = py.matmul(pxd_part.t())
                n = nx_part.contiguous().view(1, -1) * ny.view(-1, 1)
                nid = torch.argmax(d/n, 0)
                nids.append(nid)        
                
            nid = torch.cat(nids)
                
            # Old memory-inefficient code
            #d = py.matmul(pxd.t()) # nominator casted as convolution, nxm
            #n = nx.view(1, -1) * ny.view(-1, 1) # outer product nxm
            #nid = torch.argmax(d/n, 0)
            
            return px, py.index_select(0, nid)


class SemanticStyle(PatchStyle):
    '''Semantic style loss provider.

    Computes the style loss between the semantically enriched reconstruction and style image 
    as described in 
    https://arxiv.org/pdf/1603.01768.pdf
    '''

    def __init__(
        self, 
        image=None, layer_ids=None, layer_weights=None, 
        lambda_loss=1e-2, k=3, s=1,         
        semantic_style_image=None, 
        semantic_content_image=None,
        gamma=None,
        gamma_scale=0.4
        ):

        '''Create PatchStyle.

        Kwargs
        ------
        image : image
            Style image to be matched.
        layer_ids : list
            Optional list of raw network layers to compute feature losses at.
            If None defaults to [6,8,10].
        layer_weights : list
            Optional weights for each layer loss. If None uniform weights are  
            applied.
        lambda_loss : scalar
            Lagrange multiplier strengthness for this loss vs. other losses.
        k : number
            Kernel size
        s : number
            Stride
        semantic_style_image : image
            Semantic map of style
        semantic_content_image : image
            Semantic map of content
        gamma : list, scalar
            Optional gamma factor for channel concatenation per layer. If None,
            auto-tunes gamma per layer.
        gamma_scale : scale
            Weights the auto-tuned gamma factor.
        '''
        
        super(SemanticStyle, self).__init__(
            image=image, 
            layer_ids=layer_ids,
            layer_weights=layer_weights, 
            lambda_loss=lambda_loss, 
            k=k, s=s)        

        self.semantic_style_image = to_np(semantic_style_image)
        self.semantic_content_image = to_np(semantic_content_image)
        self.gamma = gamma
        self.gamma_scale = gamma_scale

    def create_loss(self, net, dev):
        return SemanticStyle.Loss(
            net, dev, self.layer_ids, self.layer_weights, self.image, self.k, self.s, 
            self.semantic_style_image, self.semantic_content_image, self.gamma, self.gamma_scale)

    def scale_by(self, f):
        image = to_image(self.image).scale_by(f) 
        sem_style = to_image(self.semantic_style_image).scale_by(f, resample=NEAREST)
        sem_cont = to_image(self.semantic_content_image).scale_by(f, resample=NEAREST)

        return SemanticStyle(
            image=self.image, 
            layer_ids=self.layer_ids, 
            layer_weights=self.layer_weights, 
            lambda_loss=self.lambda_loss, 
            k=self.k, s=self.s,
            semantic_style_image=sem_style,
            semantic_content_image=sem_cont,
            gamma=self.gamma,
            gamma_scale=self.gamma_scale)


    class Loss(PatchStyle.Loss):

        def __init__(self, net, dev, lids, w, image, k, s, sem_style, sem_cont, gamma, gamma_scale):
            super(SemanticStyle.Loss, self).__init__(net, dev, lids, w, image, k, s)
            self.sem_style = to_torch(sem_style).to(dev)
            self.sem_cont = to_torch(sem_cont).to(dev)
            self.gamma = gamma
            self.gamma_scale = gamma_scale


        def init(self):
            # Called from enter of GramStyle
            with torch.no_grad():
                self.net(self.image)
                
                # Auto-tune gamma
                if self.gamma is None:
                    self.gamma = [self.tune_gamma(a, self.sem_style) for a in self.act]
                elif isinstance(self.gamma, Iterable):
                    assert len(self.gamma) == len(self.act)
                else:
                    self.gamma = [self.gamma] * len(self.act)

                self.style_act = self.create_stack(self.sem_style, self.gamma)

                                
        def __call__(self):
            self.act = self.create_stack(self.sem_cont, self.gamma)
            return super(SemanticStyle.Loss, self).__call__()

        def create_stack(self, semantic, gamma):
            return [torch.cat((a, F.adaptive_avg_pool2d(semantic, a.shape[-2:])*g), 1) for a,g in zip(self.act, self.gamma)]

        def tune_gamma(self, a, sem):
            sem = F.adaptive_avg_pool2d(sem, a.shape[-2:])
            s = torch.norm(a, 2, 1).mean() / torch.norm(sem, 2, 1).mean()
            return s * self.gamma_scale


"""Block matrices could further improve memory efficiency of PatchStyle

x = np.random.rand(1024, 128)
y = np.random.rand(128, 2048)

def blocks(x, rows, cols):
    r = np.array_split(x, rows, 0)
    c = [np.array_split(rr, cols, 1) for rr in r]
    return np.array(c)

def mul(bx, by):
    a = bx.shape[0]
    s = bx.shape[1]    
    b = by.shape[1]
    
    f = np.zeros((bx.shape[0]*bx.shape[2], by.shape[1]*by.shape[3]))
    
    
    for aa in range(a):
        for bb in range(b):
            cb = f[aa*bx.shape[2]:(aa+1)*bx.shape[2], bb*by.shape[3]:(bb+1)*by.shape[3]]
            for ss in range(s):
                cb += np.dot(bx[aa,ss], by[ss,bb])
    return f
    
    #return np.asarray(c)


bx = blocks(x, 4, 2)
by = blocks(y, 2, 4)

print(bx.shape, by.shape)

(abs(mul(bx, by) - np.dot(x, y))).sum()
"""