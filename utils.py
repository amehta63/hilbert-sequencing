import torch
import torch.nn.functional as functional
import datetime
from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np

def getNowString():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M"))

def hilbertIndexTensor(p=5):
    size = 2**p
    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(size**2))
    points = hilbert_curve.points_from_distances(distances)
    indexes = torch.arange(2**(2*p)).view(size,size)
    for i, point in enumerate(points):
        indexes[point[0],point[1]] = i
    return indexes

def hilbertCurveAnyD(x, dim=1, pad=0): #TODO make it actually anyD, right now it will only work for batch x sequence x embedding
    original_shape = x.shape
    p = (int(np.ceil(np.sqrt(x.shape[dim])))-1).bit_length() 
    size = 2**(p)
    length = 2**(2*p)
    dims = [i for i in range(len(x.shape)) if i != dim]
    dims.append(dim)
    x = torch.permute(x, tuple(dims))
    x = functional.pad(x, (pad, length-x.shape[-1]))
    full_shape = x.shape
    newdims = [i for i in range(len(x.shape))]
    newdims.insert(dim, newdims.pop())
    x = torch.permute(x, tuple(newdims))

    indexes = hilbertIndexTensor(p=p).view(-1).expand(x.size(0), -1) #batch it up
    hilbert_x = torch.gather(x, dim=dim, index=indexes.unsqueeze(-1).expand(-1, -1, x.size(2)))  # [64, 1024, 1280]
    return hilbert_x.view(*full_shape[0:dim], size, size, *full_shape[dim:-1])
    
def hilbertCurve1Dto2D(x, dim=0):
    """
    This is actually the same as athe 2D to 3D version but due to the style of padding, it is size limited.
    I might fix that later and delete this method.
    """
    x = x.squeeze()
    if len(x.shape) > 1:
        raise ValueError('1Dto2D method recieved more than 1D input.')
    # calculate the minimum square side length that is a power of 2 that will hold all of the data
    p = (int(np.ceil(np.sqrt(len(x))))-1).bit_length() 
    # 1D -> 2D means n=2
    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(len(x)))
    points = hilbert_curve.points_from_distances(distances)
    y = x.clone()
    y = functional.pad(y, (0, 2**(2*p)-len(x)))
    y = y.reshape(2**p, 2**p, -1)
    for point, dist in zip(points, distances):
        if dist < len(x): y[point[0], point[1]] = x[dist]
        else: y[point[0], point[1]] = torch.zeros(x[0].shape)
    return y

def hilbertCurve2Dto3D(x, dim=0):
    """
    A 2D to 3D hilbert curve function. This method takes a 2D array and uses the first dimension to wrap the array into
    a hilbert 2D curve. This can be imagined as taking a sheet of data, and rolling the sheet so that the cross section
    looks like the fractal hilbert curve.

    Args:
        x: a two dimensional torch tensor that will be wrapped according to the vector x[0]
        dim: dimensions of x, originally meant to indicate the level of curvature, but is no longer used. Will delete in future.

    Returns:
        A 3D array where for each x[i,:] there exists a returned x_new[j,k,:] where all data in dimensions ':' is the same.
    """
    x = x.squeeze()
    if len(x.shape) != 2:
        raise ValueError('2Dto3D method recieved non-2D input.')
    # calculate the minimum square side length that is a power of 2 that will hold all of the data
    p = (int(np.ceil(np.sqrt(len(x))))-1).bit_length() 
    # 2D -> 3D on only one dimension means n=2
    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(2**(2*p)))
    points = hilbert_curve.points_from_distances(distances)
    y = x.clone()
    y = functional.pad(y, (0, 0, 0, 2**(2*p)-len(x)))
    all_but_first_dim = y.size()[1:]
    y = y.reshape(2**p, 2**p, *all_but_first_dim)
    for point, dist in zip(points, distances):
        if dist < len(x): y[point[0], point[1], :] = x[dist, :]
        else: y[point[0], point[1], :] = torch.zeros_like(x[0, :]).bfloat16()
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
