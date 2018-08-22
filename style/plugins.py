# py-style-transfer
# Copyright 2018 Christoph Heindl.
# Licensed under MIT License
# ============================================================

import math
import style.image as image
import torch.nn.functional as F
import numpy as np

from PIL import Image

class Plugin:
    '''An optimization plugin base class.'''

    def prepare(self, p, a, x, **kwargs):
        pass

    def after_loss(self, x, loss):
        return loss

    def after_backward(self, x):
        pass

    def after_step(self, x):
        pass
    

class SeamlessPlugin(Plugin):
    '''Changes optimization to generate images that seamless tile.'''

    def __init__(self, image_shape, border):   
        self.f = min(image_shape[:2]) / border

    def prepare(self, cl, sl, x, **kwargs):
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
