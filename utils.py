import torch
import torch.nn.functional as functional
import datetime
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np

def getNowString():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M"))

def hilbertCurve(x, dim=0):
    p = (int(np.ceil(np.sqrt(len(x))))-1).bit_length()
    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(len(x)))
    points = hilbert_curve.points_from_distances(distances)
    y = x.clone()
    if len(y.shape) > 1:
        y = y.T
    y = functional.pad(y, (0, 2**(2*p)-len(x)))
    y = y.reshape(2**p, 2**p, -1)
    for point, dist in zip(points, distances):
        if dist < len(x): y[point[0], point[1]] = x[dist]
        else: y[point[0], point[1]] = torch.zeros(x[0].shape)
    return y

def save_model(model=None, save_pth=None, epoch=None, index=None, loss=None):
    if model is None:
        raise Exception("must include a model")
    elif save_pth is None:
        raise Exception("must include a save path")
    elif epoch is None or index is None or loss is None:
        print("saving full model " + save_pth)
        torch.save(model, save_pth)
    else:
        print("saving partial model " + save_pth)
        torch.save({
            'epoch': epoch,
            'batch_index': index,
            'model_state_dict': model.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, save_pth)
