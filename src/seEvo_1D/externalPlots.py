import scipy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os

def readPaths(file_path):
    path = file_path.split("/")
    out = ""
    name = ""
    number = ""
    
    for i in path:
        if i.endswith(".npz"):
            i = i.split('_')
            for x in i:
                if x.endswith(".npz"):
                    number = x.rstrip(".npz")
                else:
                    name = name + x + '_'
            name.lstrip('_')
        else:
            out = out + i + '/'
    out = out + "Figures/" + number + "/"
    return [file_path, out, name, number]        
    
def mutationWave(file_path):
    print(file_path)
    for x in file_path:
        _in, _out, _name, _num = readPaths(x)
        fig = plt.figure(figsize=(40,20))
        ax = fig.add_subplot() 
        pop = sc.sparse.load_npz(_in)
        popSize = 0
        
        if 'analytical' in _name:
            popSize = sum(pop[:,1].toarray())
            ax.plot(pop[:,0].toarray(), pop[:,1].toarray(), 'k', linewidth='10')
            
        elif 'normal' in _name:                    
            popSize = pop._shape[0]
            ax.hist(pop[:,1].toarray(), color='red')
            
        ax.legend(prop={'size':30})
        
        ax.set_xlabel("Mutation number", labelpad=50, fontdict={'fontsize':50})
        ax.set_ylabel("Cells", labelpad=50, fontdict={'fontsize':50})
        ax.set_title("Mutation Wave, Population: %i" % (popSize), pad=50, fontdict={'fontsize':70})
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(40) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(40) 
        
        try:
            os.makedirs(_out, exist_ok=True) 
        except OSError as error:
            print(error)
        finally:
            if os.path.exists(_out + "%s_mutation_wave.jpg" % _num):
                os.remove(_out + "%s_mutation_wave.jpg" % _num)
            fig.savefig(_out + "%s_mutation_wave.jpg" % _num)
            plt.close(fig)            
    
def fitnessWave(file_path):
    for x in file_path:
        _in, _out, _name, _num = readPaths(x)
        fig = plt.figure(figsize=(40,20))
        ax = fig.add_subplot() 
        pop = sc.sparse.load_npz(_in)
        popSize = 0
        
        if 'analytical' in _name:
            print('same as mutation wave')
            continue
        
        elif 'normal' in _name:        
            popSize = pop._shape[0]            
            ax.hist(pop[:,0].toarray(), color='red')
            
        ax.legend(prop={'size':30})
        
        ax.set_xlabel("Fitness", labelpad=50, fontdict={'fontsize':50})
        ax.set_ylabel("Cells", labelpad=50, fontdict={'fontsize':50})
        ax.set_title("Fitness Wave, Population: %i" % (popSize), pad=50, fontdict={'fontsize':70})
        
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(40) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(40) 
        
        try:
            os.makedirs(_out, exist_ok=True)
        except OSError as error:
            print(error)
        finally:
            if os.path.exists(_out + "%s_fitness_wave.jpg" % _num):
                os.remove(_out + "%s_fitness_wave.jpg" % _num)
            fig.savefig(_out + "%s_fitness_wave.jpg" % _num)
            plt.close(fig)            

def combainedMutWave(file_path):
    _out = ''
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_subplot() 
    ax.clear()
    for x in file_path:
        _in, _out, _name, _num = readPaths(x)
        
        pop = sc.sparse.load_npz(_in)
        popSize = 0
        
        if 'analytical' in _name:
            ax.plot(pop[:,0].toarray(), pop[:,1].toarray(), 'k', linewidth='10')
        
        elif 'normal' in _name:       
            temp = pop.toarray()
            ax.hist(temp[:,1],  color = "red")
                    
    ax.set_xlabel("Generation", labelpad=50, fontdict={'fontsize':50})
    ax.set_ylabel("Population size", labelpad=50, fontdict={'fontsize':50})
    ax.set_title("Combined mutation wave", pad=50, fontdict={'fontsize':70})
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(40) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)  
    
    try:
        os.makedirs(_out, exist_ok=True) 
    except OSError as error:
        print(error)
    finally:
        if os.path.exists(_out + "combined_mutation_wave.jpg"):
            os.remove(_out + "combined_mutation_wave.jpg")
        fig.savefig(_out + "combined_mutation_wave.jpg")
        plt.close(fig)  

def popGrowth(file_path):
    data = []
    _out = ''
    for x in file_path:
        _in, _out, _name, _num = readPaths(x)
        
        pop = sc.sparse.load_npz(_in)
        popSize = 0
        
        if 'analytical' in _name:
            popSize = int(sum(pop[:,1].toarray())[0])
        
        elif 'normal' in _name:       
            popSize = pop._shape[0]
        
        data.append([_num, popSize])
        
    df = pd.DataFrame(data)
    df.columns = ["id","size"]
    
    ax = df.plot.line(x='id', y='size', figsize=(40,20), linewidth='10')
    
    ax.set_xlabel("Generation", labelpad=50, fontdict={'fontsize':50})
    ax.set_ylabel("Population size", labelpad=50, fontdict={'fontsize':50})
    ax.set_title("Population growth", pad=50, fontdict={'fontsize':70})
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(40) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(40) 
    
    fig = ax.get_figure()    
    
    try:
        os.makedirs(_out, exist_ok=True) 
    except OSError as error:
        print(error)
    finally:
        if os.path.exists(_out + "population_growth.jpg"):
            os.remove(_out + "population_growth.jpg")
        fig.savefig(_out + "population_growth.jpg")
        plt.close(fig)  