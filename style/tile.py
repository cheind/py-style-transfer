from tqdm import tqdm
import numpy as np

from style.image import borderless_view

class TiledGeneration:
    
    def __init__(self, st):
        self.st = st
        
    def generate(self, seed, a, c_id, style_weights, grid_shape, border=32, **kwargs):
        
        final_shape = seed.shape
        tile_shape = (final_shape[0]//grid_shape[0], final_shape[1]//grid_shape[1], 3)
         
        seed = np.pad(seed, ((border,border),(border,border),(0,0)), 'reflect')
        unused = np.zeros_like(seed)
        final = np.zeros(final_shape, np.float32)
        
        with tqdm(total=grid_shape[0]*grid_shape[1]) as t: 
            for row in range(grid_shape[0]):
                for col in range(grid_shape[1]):
                    
                    sr = tile_shape[0] * row
                    er = tile_shape[0] * (row+1) + 2*border
                    sc = tile_shape[1] * col
                    ec = tile_shape[1] * (col+1) + 2*border

                    tile,_ = self.st.run(seed[sr:er,sc:ec], a, c_id, style_weights, x=np.array(seed[sr:er,sc:ec]), disable_progress=True, **kwargs)
                    
                    sr = tile_shape[0] * row
                    er = tile_shape[0] * (row+1)
                    sc = tile_shape[1] * col
                    ec = tile_shape[1] * (col+1)
                    final[sr:er,sc:ec] = borderless_view(tile, border)
                    
                    t.update()
        
        return final