import scipy as sc
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def timereport(path, file):
    data = pd.read_csv(path + "report/" + file)
    data = data.to_numpy()
    fig = plt.figure()
    plt.plot(data)
    plt.xlabel("Generation")
    plt.ylabel("Compute time")
    plt.show()  

def basic_plotter(num, c, mutlim, path, select=0):
    # path = "E:/Simulations/Pos Neg Evolution/x5 Test/Test1/%5Ctest"
    
    y = [0 for i in range(num)]
    x = [i for i in range(num)]
    
    for i in range(num):
        a = sc.sparse.load_npz(path + "_" + str(i) + ".npz")
        y[i] = a._shape[0]
        if i % c == 0:
            temp = []
            if select:
                temp= a[:,1].toarray() + a[:,2].toarray()
            else:
                temp = a[:,1].toarray()
            fig = plt.figure()
            plt.hist(temp)
            plt.xlim(mutlim)
            plt.xlabel("Mutation number")
            plt.ylabel("Cells number")
            plt.show()
            
            # temp = a[:,0].toarray()
            # fig = plt.figure()
            # plt.hist(np.log10(temp))
            # plt.xlim(mutlim)
            # plt.xlabel("Fitness parameter [log10]")
            # plt.ylabel("Cells number")
            # plt.show()
    
    fig = plt.figure()
    plt.plot(x,y)
    plt.xlabel("Generation")
    plt.ylabel("Population size")
    plt.show()   
  
def npztotxt(path_in, fname):
    a = sc.sparse.load_npz(path_in + fname)
    df = pd.DataFrame(a.toarray())
    if a._shape[1] == 3:
        df.columns = ["fitness", "positive mutation number", "negative mutation number"]
        os.makedirs(path_in + "/CSV/", exist_ok=True)
        df.to_csv(path_in + "/CSV/" + fname.rstrip(".npz") + ".csv", index=False)
    else:
        columns = ["fitness"]
        for i in range(a._shape[0] - 1):
            columns.append(str(i))
        df.columns = columns      
        os.makedirs(path_in + "/CSV/", exist_ok=True)
        df.to_csv(path_in + "/CSV/" + fname.rstrip(".npz") + ".csv", index=False)
    
if __name__ == "__main__":
    t = 5575
    fname = "stationary_pos"
    file = "pos_report_3.txt"
    path = "E:/Simulations/Pos Neg Evolution/Stationary wave/Positive effect/"
    # timereport(path, file)
    basic_plotter(t, 10, (0,2000), path + fname)
    # npztotxt("E:/Simulations/Pos Neg Evolution/x5 Test/Test1/", "%5Ctest_1349.npz")
    # for i in range(t):
    #     name = fname + "_" + str(i) + ".npz"
    #     npztotxt(path, name)
    