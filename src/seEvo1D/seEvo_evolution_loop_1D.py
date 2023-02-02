import numpy as np
import math
import itertools
import scipy as sc
import copy
from threading import Thread
from multiprocessing import Pool

def death(iPop, pdt):
    return iPop[pdt,:]
    
def divide(iPop, pdv):
    return sc.sparse.vstack([iPop, copy.deepcopy(iPop[pdv,:])]).tocsr()
    # return np.append(iPop, copy.deepcopy(iPop[pdv,:]), axis=0)
    
def mutate(iPop, pdm, mut_effect):
    cells = len(pdm)
    if cells == 0:
        return iPop
    (a,b) = iPop._shape
    rows = copy.deepcopy(iPop[pdm,:]).toarray()
    # rows = copy.deepcopy(iPop[pdm,:])
    rows[:, 0] = rows[:, 0]*((1+abs(mut_effect))**(1 - 2*(mut_effect < 0)))
    # rows[:, 0] = rows[:, 0]/(1-abs(mut_effect))
    rows[:, 1] = rows[:, 1] + 1
    iPop = sc.sparse.vstack([iPop, rows]).tocsr()
    # iPop = np.append(iPop, rows, axis=0)
    return iPop

def seEvo1Dnorm(iPop, cap, tau_x, A, mut_prob, mut_effect, simTime):
    popSize = iPop._shape[0]
    # popSize = len(iPop[:,0])
    
    mdt = (popSize/cap)**A
    tau = tau_x * cap/popSize
    # tau = tau_x
    mdv = iPop[:,0].toarray()[:,0]
    # mdv = iPop[:,0]
    
    simTime = simTime + tau
    
    pdt = np.random.exponential(1, popSize)/mdt
    pdv = np.random.exponential(1, popSize)/mdv
    pdm = np.random.binomial(1, mut_prob, popSize)
    
    pdtx = pdt * (pdt <= tau) 
    pdtx = pdtx * (pdt < pdv)
    pdvx = pdv * (pdv <= tau) 
    pdvx = pdvx * (pdv < pdt)
    pdt = pdtx == 0
    pdv = pdvx > 0
    
    pdm = pdv & pdm
    pdm = pdm > 0
    pdv = np.logical_xor(pdv, pdm)
    
    nr = np.array(range(popSize))
    pdv = nr[pdv]
    pdt = nr[pdt]
    pdm = nr[pdm]     
    
    iPop = divide(iPop, pdv)
    
    iPop = mutate(iPop, pdm, mut_effect) 
    
    pdt = np.append(pdt, range(max(pdt)+1, iPop._shape[0]))
    # pdt = np.append(pdt, range(max(pdt)+1, len(iPop[:,0])))
    iPop = death(iPop, pdt)
    
    return iPop, simTime