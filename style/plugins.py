
import math
import style.image as image
from PIL import Image

class SeamlessPlugin:

    def __init__(self, image_shape, border):   
        self.f = min(image_shape[:2]) / border

    def prepare(self, p, a, x):
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

class FreezePlugin:

    def __init__(self, mask):   
        self.mask = mask

    def prepare(self, p, a, x):
        
        target_size = (x.shape[-1], x.shape[-2])
        source_size = self.mask.shape[:2][::-1]

        if target_size != source_size:
            self._mask = image.resize(self.mask, (x.shape[-1], x.shape[-2]), resample=Image.NEAREST)
        else:
            self._mask = self.mask            

        self._mask = x.new_tensor(image.to_torch(self._mask))
        self._p = p

    def after_backward(self, x):
        x.grad.data.mul_(1 - self._mask)

    def after_step(self, x):
        x.data.copy_(x.data * (1 - self._mask) + self._p * self._mask)
