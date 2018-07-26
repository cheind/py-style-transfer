
import style.image as image

class SeamlessTexture:

    def __init__(self, x, border):
        self.x = x
        self.border = border

    def zero_border_grads(self):
        if self.border > 0:
            outer = image.border_elements(self.x.grad.data, self.border)
            [o.zero_() for o in outer]

    def copy_content_to_border(self):
        b = self.border
        if b > 0:
            outer = image.border_elements(self.x.data, b)
            inner = image.border_elements(image.borderless_view(self.x.data, b), b)

            outer.tl.copy_(inner.br)
            outer.t.copy_(inner.fb)
            outer.tr.copy_(inner.bl)
            outer.r.copy_(inner.fl)
            outer.br.copy_(inner.tl)
            outer.b.copy_(inner.ft)
            outer.bl.copy_(inner.tr)
            outer.l.copy_(inner.fr)