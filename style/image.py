import torch
import numpy as np
import torchvision.transforms as t
from PIL import Image

def to_torch(x):
    return t.ToTensor()(x).unsqueeze(0)

def to_np(x):
    if isinstance(x, Image.Image):
        x = np.array(x, dtype=np.float32) / 255.0
    elif isinstance(x, torch.Tensor):
        x = np.transpose(x.detach().cpu().squeeze().numpy(), (1,2,0))
    else:
        x = np.asarray(x)
    return x

def to_pil(x):    
    if isinstance(x, (np.ndarray, np.generic)):
        x = (x*255).astype(np.uint8)
        x = t.ToPILImage()(x)
    elif isinstance(x, torch.Tensor):
        x = to_pil(to_np(x))    
    return x

def open(fname):
    return to_np(Image.open(fname).convert('RGB'))

def save(fname, x):
    to_pil(x).save(fname)

def resize(x, size, resample=Image.BILINEAR):
    return to_np(to_pil(x).resize(size, resample))

class ImagePyramid:

    class Scaler:
        def __init__(self, size, resample):
            self.size = size
            self.resample = resample
        
        def __call__(self, img):
            return resize(img, self.size, self.resample)

    @staticmethod 
    def get_sizes(finalsize, levels):
        def size_for_level(finalsize, level):
            s = 2 ** level
            return (finalsize[0] // s, finalsize[1] // s)
        
        return [size_for_level(finalsize, l) for l in range(levels)][::-1]

    def __init__(self, sizes, resample=Image.BILINEAR):
        self.sizes = sizes
        self.resample = resample

    def iterate(self):
        for s in self.sizes:
            yield ImagePyramid.Scaler(s, self.resample)


vgg_mean = torch.tensor([0.485, 0.456, 0.406])
vgg_std = torch.tensor([0.229, 0.224, 0.225])