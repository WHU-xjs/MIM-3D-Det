import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def save_vis_bev(features, vispath: str, visidx: int, grids: tuple=(4,4)):
    slices = grids[0] * grids[1]
    # remove bevimg, use smaller slices for getting expected balanced map
    if not os.path.exists(vispath): os.makedirs(vispath)
    # batch size locked to 1 during testing
    bevmsk = features['map'].clone().detach()[0]
    # conv encoder using relu, all positive, norm to (0,1)
    # contain few very large numbers(<3%), affecting visualization
    bevmsk = bevmsk.sqrt()
    # bevmsk = bevmsk.clamp(max=mean+3*std)
    # bevmsk = (bevmsk - mean) / (6 * std) + 0.5
    # chunk to slices, sum channels of each slice
    bevmsk = [x.sum(0) for x in bevmsk.chunk(slices, 0)]
    bevmsk = [x.to(torch.float).detach().cpu() for x in bevmsk]
    save_tensor(bevmsk, vispath+'msk-{:0>4d}.png'.format(visidx), grids)

def save_tensor(tensors, path, arrange):
    rows, cols = arrange
    grids = []
    for row in range(rows):
        grids.append(np.hstack(tensors[row*cols : row*cols+cols]))
    grids = np.vstack(grids)
    plt.figure()
    plt.imsave(path, grids, cmap=plt.get_cmap('viridis'))
    plt.close()
