import numpy as np
import torch
import torch.nn.functional as F

class LossFnc:
    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, type, value, tb):
        self.remove()

    def enter(self):
        pass

    def remove(self):
        pass

    def init(self):
        pass
        
    def __call__(self):
        raise NotImplementedError()

class LayerLoss(object):
    '''Base class for feature map based layer losses.'''

    def __init__(self, layer_ids, lambda_loss):
        self.layer_ids = layer_ids
        self.lambda_loss = lambda_loss

    def create_loss(self, net):
        raise NotImplementedError()

class ContentLoss(LayerLoss):

    def __init__(self, layer_id, lambda_loss=1e-3):
        super(ContentLoss, self).__init__([layer_id], lambda_loss)  

    def create_loss(self, net):
        return ContentLoss.Fnc(net, self.layer_ids[0])

    class Fnc(LossFnc):
        def __init__(self, net, lid):
            self.net = net
            self.lid = lid
            
        def enter(self):
            self.hook = self.net[self.lid].register_forward_hook(self.hookfn)

        def hookfn(self, n, inp, outp):
            self.act = outp

        def remove(self):
            self.hook.remove()

        def init(self):
            # assumes net(content) called
            self.ref = self.act.data.clone()
            
        def __call__(self):
            # assumes net(x) called
            return F.mse_loss(self.act, self.ref)


class GramStyleLoss(LayerLoss):

    def __init__(self, layer_ids, layer_weights=None, lambda_loss=1e4):        
        super(GramStyleLoss, self).__init__(layer_ids, lambda_loss)
        self.layer_weights = layer_weights

    def create_loss(self, net):
        return GramStyleLoss.Fnc(net, self.layer_ids, self.layer_weights)

    class Fnc(LossFnc):
        def __init__(self, net, lids, w):
            self.lids = lids
            self.w = w
            self.net = net

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

        def hookfn(self, n, inp, outp):
            self.act.append(outp)

        def prehookfn(self, n, inp):
            self.act = []

        def remove(self):
            [h.remove() for h in self.hooks]
            self.prehook.remove()

        def init(self):
            # assumes net(a) called     
            self.A = [self.gram(x).data.clone() for x in self.act]

        def __call__(self):
            G = [self.gram(x) for x in self.act]
            E = torch.stack([w * F.mse_loss(g, a).view(-1) for g,a,w in zip(G, self.A, self.w)])      
            return E.sum()

        def gram(self, x):
            c, n = x.shape[1], x.shape[2]*x.shape[3]
            f = x.view(c, n)
            return torch.mm(f, f.t()) / (c*n)


class PatchStyleLoss(GramStyleLoss):

    def __init__(self, layer_ids, layer_weights=None, lambda_loss=1e-2, k=3, s=1):        
        super(PatchStyleLoss, self).__init__(layer_ids, layer_weights, lambda_loss)
        self.k = k
        self.s = s

    def create_loss(self, net):
        return PatchStyleLoss.Fnc(net, self.layer_ids, self.layer_weights, self.k, self.s)

    class Fnc(GramStyleLoss.Fnc):

        def __init__(self, net, lids, w, k, s):
            super(PatchStyleLoss.Fnc, self).__init__(net, lids, w)
            self.k = k
            self.s = s

        def init(self):
            # assumes net(a) called   
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
            with torch.no_grad():
                py = F.unfold(y, ky, stride=sy).transpose(1,2).squeeze(0) # nxp
                ny = torch.norm(py, 2, 1)

            px = F.unfold(x, kx, stride=sx).transpose(1,2).squeeze(0) # mxp
            nx = torch.norm(px, 2, 1) # treat norm as constant during opt
                        
            d = py.matmul(px.t()) # nominator casted as convolution, nxm
            n = nx.view(1, -1) * ny.view(-1, 1) # outer product nxm
            
            nid = torch.argmax(d/n, 0)
            
            return px, py.index_select(0, nid)


class SemanticStyleLoss(PatchStyleLoss):

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
