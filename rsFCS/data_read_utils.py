import numpy as np
import scipy.io as sio
from registry import Registry

def readflex(filename, channels, flex=8):
    '''function to read binary data from correlator.com flex card.
    nch gives the number of channels saved in the file. flex is
    the bit mode of the file and must be 8 or 16'''
    nch = len(channels)
    if flex == 8:
        dtype_here = np.uint8
    elif flex == 16:
        dtype_here = np.uint16
    else:
        raise ValueError("flex must be 8 or 16")
    temp_data = np.fromfile(filename, dtype=dtype_here)
    if nch == 2:
        temp_data = temp_data.reshape(temp_data.size//2, 2).T
        return [temp_data[0].astype(np.uint64), temp_data[1].astype(np.uint64)]
    elif nch == 1:
        return [temp_data.astype(np.uint64)]
    else:
        raise ValueError("Problem assigning channels")

def readnpy(filename):
    temp_data = np.load(filename)
    ndim = temp_data.ndim
    if ndim == 1:
        nch = 1
        return [temp_data.astype(np.uint64)]
    elif ndim == 2:
        nch = temp_data.shape[0]
        return [temp_data[i,:].astype(np.uint64) for i in range(nch)]
    else:
        raise ValueError("Unsupported data format")


def readsav(filename, key='g'):
    datadict = sio.readsav(filename)
    temp_data = datadict[key].astype(np.uint64)
    if temp_data.ndim == 1:
        return [temp_data]
    else:
        raise ValueError("Unsupported data format")

#TODO bin, spc

READ_FXNS = {'flex' : readflex,
             'npy' : readnpy,
             'sav' : readsav}
readers = Registry()
for name, func in READ_FXNS.items():
    readers.register_object(name, func)
