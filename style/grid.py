import numpy as np

from style.image import border_elements as getb
from style.image import borderless_view as geti
from style.image import Pyramid
from style.plugins import FadeInPlugin

class GridSynthesis:    
    
    def __init__(self, st, grid_shape, tile_shape, s=16, b=32):
        self.st = st
        self.grid_shape = grid_shape
        self.tile_shape = tile_shape
        self.s = s
        self.b = b
        
    def generate(self, a, c_id, style_weights, multiscale=True, **kwargs):
        
        sizes = Pyramid.image_sizes(self.tile_shape, 4)

        seq = [None]*(self.grid_shape[0] * self.grid_shape[1])
        for row in range(self.grid_shape[0]):
            for col in range(self.grid_shape[1]):
                f = self._get_fadein(seq, row, col)
                
                if multiscale:                    
                    x,_ = self.st.run_multiscale(
                        f.content, a, c_id, style_weights, sizes,
                        x=self._random_start(a), 
                        weight_content_loss=0, 
                        plugins=[f],
                        **kwargs)
                else:
                    x,_ = self.st.run(
                        f.content, a, c_id, style_weights,
                        x=self._random_start(a), 
                        weight_content_loss=0, 
                        plugins=[f],
                        **kwargs)
                
                seq[self._index(row, col)] = x
                
        return self._create_final(seq)
        
    def _random_start(self, a):
        init = a.mean((0,1), keepdims=True)
        init = init + np.random.randn(*self.tile_shape).astype(np.float32)*1e-2
        return init        
    
    def _get_fadein(self, seq, row, col):
        
        
        left_mask = np.zeros(self.tile_shape[:2] + (1,))
        top_mask = np.zeros(self.tile_shape[:2] + (1,))
        
        w = np.linspace(0, np.pi, self.b)
        w = (1 + np.cos(w)) / 2
        
        ctx = np.zeros(self.tile_shape)
        
        if col > 0:
            bmask = getb(geti(left_mask, self.s), self.b)
            bmask.fl[:] = w.reshape(1,-1,1)
            
            left = seq[self._index(row, col-1)]
            getb(geti(ctx, self.s), self.b).fl[:] = getb(geti(left, self.s), self.b).fr[:]
        
        if row > 0: 
            bmask = getb(geti(top_mask, self.s), self.b)
            bmask.ft[:] = w.reshape(-1,1,1)
            
            top = seq[self._index(row-1, col)]
            getb(geti(ctx, self.s), self.b).ft[:] = getb(geti(top, self.s), self.b).fb[:]
            
        mask = np.maximum(left_mask, top_mask)
            
        return FadeInPlugin(ctx, mask)
        
    def _index(self, row, col):
        return self.grid_shape[1]*row + col
        
    
    def _create_final(self, seq, border=True):
        h = geti(seq[0], self.s).shape[0]
        w = geti(seq[0], self.s).shape[1]

        total_height = h * self.grid_shape[0] - self.b * (self.grid_shape[0] - 1)
        total_width = w * self.grid_shape[1] - self.b * (self.grid_shape[1] - 1)        
        final = np.empty((total_height,total_width,3))

        o = [0,0]
        for row in range(self.grid_shape[0]):
            o[1] = 0
            for col in range(self.grid_shape[1]):
                img = seq[self._index(row, col)]                
                final[o[0]:o[0]+h, o[1]:o[1]+w] = geti(img, self.s)
                o[1] += w - self.b
            o[0] += h - self.b
            
        return final