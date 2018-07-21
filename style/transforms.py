import torch
import numpy as np
import torchvision.transforms as t

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
class Clip:
    def __call__(self, tensor):
        return torch.clamp(tensor, min=0., max=1.)
    
class ToNumpy():
    def __call__(self, tensor):
        return np.transpose(tensor.numpy(), (0, 2, 3, 1))
        
normalize = t.Compose([
    t.ToTensor(), # 0..1
    t.Normalize(mean=mean, std=std)    
])

denormalize = t.Compose([
    Denormalize(mean=mean, std=std),
    Clip()    
])

def to_np_image(x, squeeze=True):
    x = denormalize(x.detach().cpu())
    x = ToNumpy()(x)
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