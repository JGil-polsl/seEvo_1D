import numpy as np
import time
import os
import math
import copy
import scipy as sc
import pandas as pd
from pathlib import Path  
from threading import Thread

from seEvo_evolution_loop_1D import seEvo1Dnorm
from seEvo_analytical_model_1D import seEvo1Danalytical
# from seEvo1D.seEvo_evolution_loop_1D import seEvo1Dnorm
# from seEvo1D.seEvo_analytical_model_1D import seEvo1Danalytical

end = False

def saveToFile(df, file_localization, file_name, iter_outer):    
    try:
        os.makedirs(file_localization, exist_ok=True) 
    except OSError as error:
        print(error)
    finally:
        sc.sparse.save_npz(file_localization + "/" + file_name + "_" + str(iter_outer), df)

def commands(q, ID, iPop, file_localization, file_name, iter_outer, skip, iter_inner, cycle, tau, select):
    global end
    queue_data = q.get()
    if(queue_data[0] == '1' and queue_data[1] == str(ID)):
        if(queue_data[2] == "exit"):
            print("exit")
            end = True
    else:
        q.put(queue_data)
 
    
def plotter(iPop, file_name, file_localization, iter_outer, plots, select):
    if plots & 1:
        print("TODO")
    if plots & 2:
        print("TODO")
    if plots & 16: 
        if select == 0:
            saveToFile(iPop, file_localization, file_name + '_normal', iter_outer)
        elif select == 1:
            saveToFile(sc.sparse.csr_matrix(iPop.T), file_localization, file_name + '_analytical', iter_outer)
        
    
def seEvoInit(iPop, 
              params, 
              file_name="", 
              file_localization="", 
              plots=0, q=None, ID=0, select=0, break_type=0):
    global end

    cap = params[1]
    steps = params[2]
    tau = params[3]
    skip = params[4]
    mut_effect = params[6]
    mut_prob = params[5]
    A = params[7]
    
    simTime = 0    
    iter_inner = 0
    iter_outer = 0
    cycle = round(skip/tau)
    
    t = time.time()
    tx = time.time()
    clear = False
    
    while 1:   
        if q != None:
            if not q.empty():
                commands(q, ID, iPop, file_localization, file_name, iter_outer, skip, iter_inner, cycle, tau, select)

        if iter_outer <= simTime:
            begin = 0
            t = time.time() - t  
            if select == 0:
                q.put(['0', str(ID), str(iter_outer)])
            elif select == 1 and iter_outer % 10 == 0:
                q.put(['0', str(ID), str(iter_outer)])
            
            if iter_inner * skip <= simTime:
                plotter(copy.deepcopy(iPop), file_name, file_localization, copy.copy(iter_inner * skip), plots, select)                
                iter_inner = iter_inner + 1
 
            clear = True
            
            if plots == 16:
                tx = time.time() - tx
                if not os.path.exists(file_localization + '/' + "report/"  + file_name + "_report_" + str(ID) + ".txt"):
                    os.makedirs(file_localization + '/' + "report/", exist_ok=True)
                    FILE = open(file_localization + '/' + "report/"  + file_name + "_report_" + str(ID) + ".txt", 'w')
                    FILE.write("name: %s" % file_name)
                    FILE.write('\n')
                    FILE.write(str(ID) + ',' + str(tx))
                    FILE.write('\n')
                    FILE.close()
                else:
                    FILE = open(file_localization + '/' + "report/"  + file_name + "_report_" + str(ID) + ".txt", 'a')
                    FILE.write(str(ID) + ',' + str(tx))
                    FILE.write('\n')
                    FILE.close()
                tx = time.time()            
            t = time.time()
            
            iter_outer = int(simTime) + 1
        
        if end:
            q.put(['exit', str(ID) + ', analytical' * (select == 1) + ', normal' * (select == 0)])
            break
        
        if not select:
            if (int(simTime) - steps >= 0 or iPop._shape[1] >= 10**6) and break_type == 0:
                print('all steps')
                end = True
            elif (iPop._shape[1] >= steps or iPop._shape[1] >= 10**6) and break_type == 1 and select != 2:
                print('all cells')
                end = True
        else:
            if (int(simTime) - steps >= 0 or sum(iPop[1,:]) >= 10**6) and break_type == 0:
                print('all steps')
                end = True
            elif (sum(iPop[1,:]) >= steps or sum(iPop[1,:]) >= 10**6) and break_type == 1 and select != 2:
                print('all cells')
                end = True
        
        if select == 0:            
            iPop, simTime = seEvo1Dnorm(iPop, cap, tau, A, mut_prob, mut_effect, simTime)
        elif select == 1:            
            iPop, simTime = seEvo1Danalytical(iPop, cap, tau, A, mut_prob, mut_effect, simTime)
            
        resume = 0
             