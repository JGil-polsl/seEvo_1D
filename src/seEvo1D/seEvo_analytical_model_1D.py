import numpy as np
import math
import itertools
import scipy as sc
import copy
from threading import Thread
from multiprocessing import Pool

mdv = [1]

def mf_dif(iPop, mdt, pdm, mut_effect):
    global mdv
    alfa = 1
    # Nfp = (1 - np.exp(-alfa * iPop))
    Nfp = iPop > 1
    A = -mdt * iPop
    A = A + (1 - pdm) * mdv * iPop * Nfp
    A[1:len(iPop)] = A[1:len(iPop)] + pdm * mdv[0:len(mdv)-1] * iPop[0:len(iPop)-1] * Nfp[0:len(Nfp)-1]
    
    return A
    
def rk4(iPop, tau, mdt, pdm, mut_effect):
    k1 = tau*mf_dif(iPop[1,:], mdt, pdm, mut_effect)
    k2 = tau*mf_dif(iPop[1,:]+k1/2, mdt, pdm, mut_effect)
    k3 = tau*mf_dif(iPop[1,:]+k2/2, mdt, pdm, mut_effect)
    k4 = tau*mf_dif(iPop[1,:]+k3, mdt, pdm, mut_effect)
    iPop[1,:] = iPop[1,:]+(1/6)*(k1+2*k2+2*k3+k4)

def seEvo1Danalytical(iPop, cap, tau_x, A, mut_prob, mut_effect, simTime):
    global mdv
    popSize = sum(iPop[1,:])
    mdt = (popSize/cap)**A
    tau = tau_x * cap/popSize
    # tau = tau_x
    simTime = simTime + tau
    
    if iPop[1,len(iPop[0,:]) - 1] > 0:
        # mdv = np.append(mdv, math.exp(mut_effect * (mdv[len(mdv)-1] + 1)))
        c = np.array(range(int(iPop[0, len(iPop[0,:])-1])+1, int(iPop[0, len(iPop[0,:])-1])+501, 1))
        # mdv = np.append(mdv, np.exp(mut_effect * c))
        mdv = np.append(mdv, ((1+abs(mut_effect))**c)**(1-2*(mut_effect < 0)))
        a = len(iPop[0,:])
        iPop = np.array([np.append(iPop[0,:], np.zeros((500))), np.append(iPop[1,:], np.zeros((500)))])
        iPop[0,a:len(iPop[1,:])] = c
                
    rk4(iPop, tau, mdt, mut_prob, mut_effect)
    
    if iPop[1,0] < 1:
        mdv = mdv[1:len(iPop[0,:])]
        iPop = iPop[:,1:len(iPop[0,:])]
    
    return iPop, simTime    