import numpy as np

def wmean(x,w):
    retVal = 0
    ld = 0
    for xx, ww in np.nditer([x,w]):
        retVal = retVal + xx*ww
        ld = ld + ww
    
    return retVal/ld