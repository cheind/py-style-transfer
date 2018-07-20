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
        t = np.transpose(tensor.detach().numpy(), (1, 2, 0))
        t *= 255.
        return t.astype(np.uint8)

normalize = t.Compose([
    t.ToTensor(), # 0..1
    t.Normalize(mean=mean, std=std)    
])

denormalize = t.Compose([
    Denormalize(mean=mean, std=std),
    Clip(),
    ToNumpy()    
])