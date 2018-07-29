import torch
import numpy as np
import math
import torchvision.transforms as t
from collections import namedtuple
from PIL import Image

vgg_mean = torch.tensor([0.485, 0.456, 0.406])
vgg_std = torch.tensor([0.229, 0.224, 0.225])

def to_torch(x):
    return t.ToTensor()(x).unsqueeze(0)

def to_np(x):
    if isinstance(x, Image.Image):
        x = np.array(x, dtype=np.float32) / 255.0
        if x.ndim == 2:
            x = np.expand_dims(x, -1)
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze().numpy()
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        x = np.transpose(x, (1,2,0))
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

BILINEAR = Image.BILINEAR
NEAREST = Image.NEAREST

def resize(x, size, resample=BILINEAR):
    if isinstance(x, (np.ndarray, np.generic)):
        if x.shape[:2][::-1] == size:
            return x
    return to_np(to_pil(x).resize(size, resample))

def noisy(x, mean=0, std=1e-2):
    n = np.random.normal(mean, std, size=x.shape).astype(np.float32)*std + mean
    return np.clip(x+n, 0, 1)

def rotate(x, degree):
    return to_np(to_pil(x).rotate(degree))


def borderless_view(x, border):    
    b = border
    if b > 0:
        if isinstance(x, torch.Tensor):
            return x[...,b:-b, b:-b]
        else:
            return x[b:-b, b:-b]
    else:
        return x

class Pyramid:

    class Scaler:
        def __init__(self, size, resample):
            self.size = size
            self.resample = resample
        
        def __call__(self, img):
            return resize(img, self.size, self.resample)

    @staticmethod 
    def image_sizes(finalshape, levels):
        finalsize = finalshape[:2][::-1]
        
        def size_for_level(level):
            s = 2 ** level
            return (finalsize[0] // s, finalsize[1] // s)
        
        return [size_for_level(l) for l in range(levels)][::-1]

    @staticmethod
    def scaled_border_sizes(sizes, border):
        if border > 0:
            mins = [min(s) for s in sizes]
            f = mins[-1] / border
            return [int(max(math.ceil(s/f),0)) for s in mins]
        else:
            return [0]*len(sizes)

    def __init__(self, sizes, resample=Image.BILINEAR):
        self.sizes = sizes
        self.resample = resample

    def iterate(self):
        for s in self.sizes:
            yield Pyramid.Scaler(s, self.resample)



Border = namedtuple('Border', 'tl t tr r br b bl l ft fr fb fl')

def border_elements(x, b):    

    if isinstance(x, torch.Tensor):
        h, w = x.shape[-2:]
        
        return Border(
            tl=x[...,:b, :b],
            t=x[...,:b, b:-b],
            tr=x[...,:b, -b:],
            r=x[...,b:-b, -b:],
            br=x[...,-b:, -b:],
            b=x[...,-b:, b:-b],
            bl=x[...,-b:, :b],
            l=x[...,b:-b, :b],
            ft=x[...,:b, :],
            fr=x[...,-b:],
            fb=x[...,-b:, :],
            fl=x[...,:b],
        )
    else:
        # assume numpy
        h, w = x.shape[:2]
        
        return Border(
            tl=x[:b, :b],
            t=x[:b, b:-b],
            tr=x[:b, -b:],
            r=x[b:-b, -b:],
            br=x[-b:, -b:],
            b=x[-b:, b:-b],
            bl=x[-b:, :b],
            l=x[b:-b, :b],
            ft=x[:b, :],
            fr=x[:,-b:],
            fb=x[-b:, :],
            fl=x[:,:b],
        )