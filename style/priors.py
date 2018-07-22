import torch

def tv_prior(x, beta):
    '''Compute the total variation'''

    tl = x[..., :-1, :-1]
    tr = x[..., :-1,  1:]
    bl = x[..., 1:,  :-1]
    
    tv = ((tl - tr)**2 + (tl - bl)**2)

    # If beta is 1 and tv is zero, than the gradient is inf.
    tv = tv + 1e-6   
    tv = tv **(beta*0.5)

    return tv.mean()
