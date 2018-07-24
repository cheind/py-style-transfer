import torch
import numpy as np
import torch.nn
import torchvision.transforms as t
import math
from PIL import Image

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


class Normalize(torch.nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

        self.mean = torch.nn.Parameter(mean.view(1,3,1,1))
        self.std = torch.nn.Parameter(std.view(1,3,1,1))

    def forward(self, x):
        return (x - self.mean) / self.std
    

to_tensor = t.ToTensor()

def to_np_image(x, squeeze=True):
    x = np.transpose(x.detach().cpu().numpy(), (0, 2, 3, 1))
    if squeeze and x.ndim == 4 and x.shape[0] == 1:
        x = np.squeeze(x, 0)
    return x

def to_pil_image(x):    
    x = to_np_image(x)
    x = (x*255).astype(np.uint8)
    if x.ndim == 3:
        x = t.ToPILImage()(x)
    else:
        x = [t.ToPILImage()(x[i]) for i in range(x.shape[0])]
    return x    


class ImagePyramid:

    class Scaler:
        def __init__(self, size, resample):
            self.size = size
            self.resample = resample
        
        def __call__(self, img):
            if not isinstance(img, Image.Image):
                img = to_pil_image(img)
            return img.resize(self.size, self.resample)

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
