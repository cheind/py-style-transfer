
import math
import style.image as image
import torch.nn.functional as F
import numpy as np

from PIL import Image

class Plugin:
    def prepare(self, p, a, x, **kwargs):
        pass

    def after_loss(self, x, loss):
        return loss

    def after_backward(self, x):
        pass

    def after_step(self, x):
        pass
    

class SeamlessPlugin(Plugin):

    def __init__(self, image_shape, border):   
        self.f = min(image_shape[:2]) / border

    def prepare(self, p, a, x, **kwargs):
        s = min(x.shape[-2:])
        self._border = int(max(math.ceil(s/self.f), 1))

    def after_backward(self, x):
        outer = image.border_elements(x.grad.data, self._border)
        [o.zero_() for o in outer]

    def after_step(self, x):
        outer = image.border_elements(x.data, self._border)
        inner = image.border_elements(image.borderless_view(x.data, self._border), self._border)

        outer.tl.copy_(inner.br)
        outer.t.copy_(inner.fb)
        outer.tr.copy_(inner.bl)
        outer.r.copy_(inner.fl)
        outer.br.copy_(inner.tl)
        outer.b.copy_(inner.ft)
        outer.bl.copy_(inner.tr)
        outer.l.copy_(inner.fr)

class FadeInPlugin(Plugin):
    def __init__(self, content, mask):   
        self.mask = mask
        self.content = content

    def prepare(self, p, a, x, **kwargs):        
        target_size = (x.shape[-1], x.shape[-2])
        source_size = self.mask.shape[:2][::-1]

        if target_size != source_size:
            self._mask = image.resize(self.mask, (x.shape[-1], x.shape[-2]), resample=image.NEAREST)
            self._content = image.resize(self.content, (x.shape[-1], x.shape[-2]), resample=image.BILINEAR)
        else:
            self._mask = self.mask    
            self._content = self.content  

        self._mask = x.new_tensor(image.to_torch(self._mask))
        self._content = x.new_tensor(image.to_torch(self._content))
        
        self._iter = 0
        
        #self._s = 0.001
        #n = kwargs['niter']
        #self._k = -np.log(self._s)/n
        self._k = 1 / kwargs['niter']

    def after_backward(self, x):
        self._iter += 1
        #f = min(self._k * self._iter, 0.2)
        f=0.25
        #f = self._s*np.exp(self._k*self._iter)

        mix = x.data * (1 - f) + self._content * f
        x.data.copy_(x.data * (1 - self._mask) + mix * (self._mask))



class ReconstructPlugin(Plugin):

    def __init__(self, mask, max_alpha):   
        self.mask = mask
        self.max_alpha = max_alpha

    def prepare(self, p, a, x, **kwargs):        
        target_size = (x.shape[-1], x.shape[-2])
        source_size = self.mask.shape[:2][::-1]

        if target_size != source_size:
            self._mask = image.resize(self.mask, (x.shape[-1], x.shape[-2]), resample=Image.NEAREST)
        else:
            self._mask = self.mask      

        self._mask = x.new_tensor(image.to_torch(self._mask))
        
        self._p = p.detach()
        self._iter = 0
        self._k = math.log(self.max_alpha) / kwargs['niter']

    def after_loss(self, x, loss):
        err = (self._mask * (self._p - x))**2
        err = err.sum() / self._mask.sum()        
        loss = loss +  err * math.exp(self._k*self._iter)

        self._iter += 1

        return loss
