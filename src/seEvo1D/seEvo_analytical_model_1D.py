import numpy as np
import math
import itertools
import scipy as sc
import copy
from threading import Thread
from multiprocessing import Pool

def mdv(mut_num, mut_effect):
    if mut_effect > 0:
        return (1+mut_effect)**mut_num
    else:
        return 1/((1-mut_effect)**mut_num)

def mf_dif(iPop, mdt, pdm, mut_effect):
    alfa = 1
    nPop = np.zeros((iPop._shape[0], iPop._shape[1]))
    nPop[:,0] = np.array([x for x in range(int(iPop[0,0]),int(iPop[iPop._shape[0]-1,0])+1)])
    A = - mdt * iPop[:,1].toarray()
    B = (1 - pdm) * mdv(iPop[:,0].toarray(), mut_effect) * iPop[:,1].toarray() * (iPop[:,1].toarray() >= 1)
    C = np.zeros((iPop._shape[0], 1))
    C[1:len(C)] = mdv(iPop[0:iPop._shape[0]-1,0].toarray(), mut_effect) * pdm * iPop[0:iPop._shape[0]-1,1].toarray() * (iPop[0:iPop._shape[0]-1,1].toarray() >= 1)
    nPop = sc.sparse.csr_matrix(nPop)
    nPop[:,1] = A + B + C 
  
    return nPop
    
def rk4(iPop, tau, mdt, pdm, mut_effect):
    k1 = sc.sparse.csr_matrix(np.zeros((iPop._shape[0], iPop._shape[1])))
    k2 = sc.sparse.csr_matrix(np.zeros((iPop._shape[0], iPop._shape[1])))
    k3 = sc.sparse.csr_matrix(np.zeros((iPop._shape[0], iPop._shape[1])))
    k4 = sc.sparse.csr_matrix(np.zeros((iPop._shape[0], iPop._shape[1])))
    k1[:,1] = tau*mf_dif(iPop, mdt, pdm, mut_effect)[:,1]
    k2[:,1] = tau*mf_dif(iPop+k1/2, mdt, pdm, mut_effect)[:,1]
    k3[:,1] = tau*mf_dif(iPop+k2/2, mdt, pdm, mut_effect)[:,1]
    k4[:,1] = tau*mf_dif(iPop+k3, mdt, pdm, mut_effect)[:,1]
    iPop = iPop+(1/6)*(k1+2*k2+2*k3+k4)
    return iPop

def seEvo1Danalytical(iPop, cap, tau_x, A, mut_prob, mut_effect, simTime):
    popSize = int(sum(iPop[:,1].toarray())[0])
    mdt = (popSize/cap)**A
    tau = tau_x * cap/popSize
    simTime = simTime + tau
    
    if iPop[iPop._shape[0]-1, 1] >= 1:
        iPop = sc.sparse.vstack([iPop, [iPop[iPop._shape[0]-1,0] + 1, 0]]).tocsr()
        
    iPop = rk4(iPop, tau, mdt, mut_prob, mut_effect)
    
    if iPop[0,1] < 1:
        iPop = iPop[1:iPop._shape[0]-1,:]
    
    return iPop, simTime
    