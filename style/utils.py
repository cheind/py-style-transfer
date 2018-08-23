import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np

def gallery(imgtuples, rows=1, cols=None, figsize=None):
    cols = cols or int(math.ceil(len(imgtuples) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = np.asarray(axs).flatten()
    
    for idx, (k,v) in enumerate(imgtuples):
        axs[idx].set_title(k)
        axs[idx].axis('off')
        axs[idx].imshow(v)

    return fig

def animate_progress(g, shape, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    img = ax.imshow(np.zeros(shape, np.float32))
    ax.set_axis_off()

    def updateimg(x):
        img.set_data(x)
        return img,

    return animation.FuncAnimation(fig, updateimg, frames=g, interval=100, blit=False)

def show_progress_ipython(g, shape, figsize=None):    
    from IPython.display import display, clear_output

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(np.zeros(shape, np.float32))
    ax.set_axis_off()

    for x in g:
        clear_output(wait=True)
        img.set_data(x)
        display(fig)
    clear_output(wait=True)