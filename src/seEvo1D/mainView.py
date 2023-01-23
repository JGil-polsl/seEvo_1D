import PyQt5.QtWidgets as qtWidget
import PyQt5.QtGui as qtGui
from PyQt5.QtCore import Qt
import sys
import numpy as np
import scipy as sc
import pandas as pd
from pathlib import Path
import copy
import re

from threading import Thread
from multiprocessing import Process, Queue

# from seEvo_evolution_init import seEvoInit
# import externalPlots as externalPlots
from seEvo1D.seEvo_evolution_init import seEvoInit
import seEvo1D.externalPlots as externalPlots

disclaimer = '''
    seEvo - slightly effect evolution simulator basing on Gillespie algorithm.
    Copyright (C) 2023 by Jarosław Gil. 
    This program comes with ABSOLUTELY NO WARRANTY.
    This is free software, and you are welcome to redistribute it
    under certain conditions.
    \n
    Software can provide multiple simulation - start simulation button starts new process.
    \n
    Example usage:
        1. Set file name and path on general tab than select save files on output tab.
        Population matrix will be saved to '.npz' files interpreted by software with 'Skip generation save' freqency
        2. Set parameters - default values already are inserted
        3. Start simulation. On status field will appear simulation ID and updated generation number.
        4. Simulation end will be annotated. To finish simulation earlier type simulation id and push stop simulation button.
        5. After simulation push wanted output figure and select file from which it should be processed
    '''

output_about = '''
    mutation wave plot - select one or multiple files to plot mutation wave (mutation number vs cells number)
    fitness wave plot - select one or multiple files to plot fitness wave (not active for analytical model)
    population growth plot - select multiple files to plot how population grow between generations
    combined mutation wave plot - select two or more files to combine mutation wave plots on one plot (able for normal and analytical wave plot)
'''

class mainFormat(qtWidget.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.q = Queue()
        
        self.ID = 1
        self.idx_s = 0
        self.th_s = []
        self.s_ID = []
        
        self._th_pg = Process()
        self._th_cp = Process()
        self._th_mw = Process()
        self._th_fw = Process()
        self.createMainView()
        self._monitor = True
        self.th_monitor = Thread(target=self.monitor, args=(self.q, self.status))
        self.th_monitor.start()
    
    def closeEvent(self, event):
        for i in self.s_ID:
            self.q.put(['1', str(i.split(',')[0]), "exit"])
            
        for i in self.th_s:
            i.join()
            
        self._monitor = False
        self.th_monitor.join()
        event.accept()   
    
    def monitor(self, q, status):
        while self._monitor:
            if not q.empty():
                data = q.get()
                if data[0] == '0':
                    status.setText("ID: " + data[1] + ", generation: " + data[2])
                elif data[0] == 'exit':
                    for i in self.s_ID:
                        if i == data[1]:
                            idx = self.s_ID.index(i)
                            self.s_ID.remove(self.s_ID[idx])
                            self.th_s[idx].join()
                            self.th_s.remove(self.th_s[idx])
                            self.idx_s = self.idx_s - 1
                else:
                    q.put(data)
                    
            
            if self._th_mw.is_alive():
                self._th_mw.join(1)   
                if not self._th_mw.is_alive():                 
                    self.status.setText("mutation wave done")
                    # self._th_mw = None
            if self._th_fw.is_alive():
                self._th_fw.join(1) 
                if not self._th_fw.is_alive():                   
                    self.status.setText("fitness wave done")
                    # self._th_fw = None
            if self._th_pg.is_alive():
                self._th_pg.join(1) 
                if not self._th_pg.is_alive():                   
                    self.status.setText("population growth done")
                    # self._th_fw = None
            if self._th_cp.is_alive():
                self._th_cp.join(1) 
                if not self._th_cp.is_alive():                   
                    self.status.setText("combined plot done")
                    # self._th_fw = None
    
    def update(self):
        i = self.tabs.indexOf(self._tabUI)
        self.tabs.removeTab(i)
        self._tabUI = self.threadTabUI()
        self.tabs.addTab(self._tabUI, "Active threads")
    
    def about(self):
        global disclaimer
        self.showDialog(disclaimer, 'About')
            
    def fitWaveAction(self):
        if self._th_fw.is_alive():
            self.showDialog("plottin already running", 'Info')
            return
        fname = qtWidget.QFileDialog.getOpenFileNames(None, 'Open file', "Z://","Data File (*.npz)")[0] 
        if len(fname) == 0:
            self.showDialog("No file selected!", "Alert")
            return
        if all(map(lambda x: x.endswith('.npz'), fname)):
            self._th_fw = (Process(target=externalPlots.fitnessWave, args=(fname,)))
            # externalPlots.fitnessWave(fname)
            self._th_fw.start()
            self.status.setText("fitness wave plot ongoing")
        else:
            self.showDialog("Wrong file/files extension. Use only one kind at time", "Alert")
            
    def mutWaveAction(self):
        if self._th_mw.is_alive():
            self.showDialog("plottin already running", 'Info')
            return
        fname = qtWidget.QFileDialog.getOpenFileNames(None, 'Open file', "Z://","Data File (*.npz)")[0] 
        if len(fname) == 0:
            self.showDialog("No file selected!", "Alert")
            return
        if all(map(lambda x: x.endswith('.npz'), fname)):
            self._th_mw = (Process(target=externalPlots.mutationWave, args=(fname,)))
            # externalPlots.mutationWave(fname)
            self._th_mw.start()
            self.status.setText("mutation wave plot ongoing")
        else:
            self.showDialog("Wrong file/files extension. Use only one kind at time", "Alert")
            
    def popGrowthAction(self):
        if self._th_pg.is_alive():
            self.showDialog("plottin already running", 'Info')
            return
        fname = qtWidget.QFileDialog.getOpenFileNames(None, 'Open file', "Z://","Data File (*.npz)")[0] 
        if len(fname) == 0:
            self.showDialog("No file selected!", "Alert")
            return
        if all(map(lambda x: x.endswith('.npz'), fname)):
            self._th_pg = (Process(target=externalPlots.popGrowth, args=(fname,)))
            # externalPlots.popGrowth(fname)
            self._th_pg.start()
            self.status.setText("population growth plot ongoing")
        else:
            self.showDialog("Wrong file/files extension. Use only one kind at time", "Alert")
            
    def combinedMutWave(self):
        if self._th_cp.is_alive():
            self.showDialog("plottin already running", 'Info')
            return
        fname = qtWidget.QFileDialog.getOpenFileNames(None, 'Open file', "Z://","Data File (*.npz)")[0] 
        if len(fname) == 0:
            self.showDialog("No file selected!", "Alert")
            return
        if all(map(lambda x: x.endswith('.npz'), fname)):
            self._th_cp = (Process(target=externalPlots.combainedMutWave, args=(fname,)))
            # externalPlots.combainedMutWave(fname)
            self._th_cp.start()
            self.status.setText("combined mutation wave plot ongoing")
        else:
            self.showDialog("Wrong file/files extension. Use only one kind at time", "Alert")
    
    def createMainView(self):
        layout = qtWidget.QGridLayout()
        self.tabs = qtWidget.QTabWidget()
        self.tabs.addTab(self.generalTabUI(), "General")
        self.tabs.addTab(self.parametersTabUI(), "Parameters")          
        self.tabs.addTab(self.outputTabUI(), "Output")   
        self._tabUI = self.threadTabUI()
        self.tabs.addTab(self._tabUI, "Threads")     
        row = 1
        layout.addWidget(self.tabs, row, 0, 1, 5)
        
        start = qtWidget.QPushButton(self)
        start.setText("Start simulation")
        start.clicked.connect(self.simStart)
        row = row + 1
        layout.addWidget(start, row, 0, 1, 5)
        
        stop = qtWidget.QPushButton(self)
        stop.setText("Stop simulation")
        stop.clicked.connect(self.simStop)
        row = row + 1
        self._ID = qtWidget.QLineEdit(str(0))        
        self._ID.setValidator(qtGui.QIntValidator(0, 1000))
        layout.addWidget(self._ID, row, 0, 1, 2)
        layout.addWidget(stop, row, 2, 1, 3)
        
        self.status = qtWidget.QLabel()
        self.status.setText("Stopped")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFont(qtGui.QFont('Consolas', 15))
        row = row + 1  
        layout.addWidget(self.status, row, 0, 1, 5)
        
        about = qtWidget.QPushButton(self)
        about.setText("About")
        about.clicked.connect(self.about)
        row = row + 1
        layout.addWidget(about, row, 0, 1, 5)
        
        self.setLayout(layout)
        
    def generalTabUI(self):
        generalTabUI = qtWidget.QWidget()
        layout = qtWidget.QGridLayout()

        self._NONE = qtWidget.QRadioButton("Checked: Normal simulation")       
        self._analytical = qtWidget.QRadioButton("Checked: Analytical model (omit VAF)")               
        self._NONE.click()
        
        self._file_name = qtWidget.QLineEdit()
        _file_name_label = qtWidget.QLabel()
        _file_name_label.setText('File name')
        self._file_path = qtWidget.QLineEdit()
        _file_path_label = qtWidget.QLabel()
        _file_path_label.setText('File path')
        _file_path_button = qtWidget.QPushButton(self)
        _file_path_button.setText('Select path')
        _file_path_button.clicked.connect(self.selectPath)
        
        row = 0
        layout.addWidget(_file_name_label, row, 0)
        layout.addWidget(self._file_name, row, 1, 1, 2)
        row = row + 1
        layout.addWidget(_file_path_label, row, 0)
        layout.addWidget(self._file_path, row, 1)
        layout.addWidget(_file_path_button, row, 2)
        
        _group = qtWidget.QButtonGroup() 
        _group.addButton(self._NONE)
        _group.addButton(self._analytical)
        row = row + 1
        layout.addWidget(self._NONE, row, 0)
        row = row + 1
        layout.addWidget(self._analytical, row , 0) 
        
        generalTabUI.setLayout(layout)
        return generalTabUI
        
    def parametersTabUI(self):
        parametersTab = qtWidget.QWidget()
        layout = qtWidget.QGridLayout()
        
        row = 0
        self._population = qtWidget.QLineEdit(str(10**4))
        layout.addWidget(qtWidget.QLabel("Initial population, N0"), row, 0)
        layout.addWidget(self._population, row, 1)
        layout.addWidget(qtWidget.QLabel("min: 0, max: 10^7"), row, 2)
        row = row + 1
        self._capacity = qtWidget.QLineEdit(str(10**4))
        layout.addWidget(qtWidget.QLabel("Enviroment capacity, Nc"), row, 0)
        layout.addWidget(self._capacity, row, 1)
        layout.addWidget(qtWidget.QLabel("min: 0, max: 10^7"), row, 2)
        row = row + 1
        self._steps = qtWidget.QLineEdit(str(10000))
        layout.addWidget(qtWidget.QLabel("Generations (simulation steps)"), row, 0)
        layout.addWidget(self._steps, row, 1)
        layout.addWidget(qtWidget.QLabel("number of steps,(optional: 'cells' as type of break - population size)"), row, 2)
        row = row + 1
        self._tau = qtWidget.QLineEdit(str(0.005))     
        layout.addWidget(qtWidget.QLabel("Tau step, tau * Nc/N"), row, 0)
        layout.addWidget(self._tau, row, 1)
        layout.addWidget(qtWidget.QLabel("min: 0, max: 1, step: 0.001"), row, 2)
        row = row + 1
        self._skip = qtWidget.QLineEdit(str(25))
        layout.addWidget(qtWidget.QLabel("Skip generation save"), row, 0)
        layout.addWidget(self._skip, row, 1)
        layout.addWidget(qtWidget.QLabel("min: 1, max: 100, step: 1"), row, 2)
        row = row + 1
        self._mut_effect = qtWidget.QLineEdit(".0005")
        layout.addWidget(qtWidget.QLabel("Mutation effect, f/s"), row, 0)
        layout.addWidget(self._mut_effect, row, 1)
        layout.addWidget(qtWidget.QLabel("min: 0.00001, max: 0.99999"), row, 2)
        row = row + 1
        self._mut_prob = qtWidget.QLineEdit("0.025")  
        layout.addWidget(qtWidget.QLabel("Mutation probability, pf/ps"), row, 0)
        layout.addWidget(self._mut_prob, row, 1)
        layout.addWidget(qtWidget.QLabel("min: 0.00001, max: 0.99999"), row, 2)  
        row = row + 1
        self._mdt_exp = qtWidget.QLineEdit("1.0")  
        layout.addWidget(qtWidget.QLabel("Death intensity exponent, A"), row, 0)
        layout.addWidget(self._mdt_exp, row, 1)
        
        _save_params = qtWidget.QPushButton(self)
        _save_params.setText('Save Parameters')
        _save_params.clicked.connect(self.saveParams)
        row = row + 1
        layout.addWidget(_save_params, row, 0, 1, 3)
        
        _load_params = qtWidget.QPushButton(self)
        _load_params.setText('Load Parameters')
        _load_params.clicked.connect(self.loadParams)
        row = row + 1
        layout.addWidget(_load_params, row, 0, 1, 3)
        
        parametersTab.setLayout(layout)
        return parametersTab
    
    def saveParams(self):
        params, break_type = self.validateParams()
        
        try:
            if self._file_name.text() == "":
                raise Exception()
            if self._file_path.text() == "":
                raise Exception()
        except:
            self.showDialog("Enter save localization and name", "Alert")
            return
            
        dfp = pd.DataFrame(params)
        
        filepath = Path(self._file_path.text() + '/params'  + ".csv")  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        dfp.to_csv(filepath)  
    
    def loadParams(self):
        fname = qtWidget.QFileDialog.getOpenFileName(self, 'Open file', "Z://","CSV files (*.csv)")        
        df = pd.read_csv(fname[0])
        df = df.to_numpy()        
    
        self._population.setText(str(int(df[0,1])))
        self._capacity.setText(str(int(df[1,1])))
        self._steps.setText(str(int(df[2,1])))
        self._tau.setText(str(df[3,1]))
        self._skip.setText(str(df[4,1]))
        self._mut_prob.setText(str(df[5,1]))
        self._mut_effect.setText(str(df[6,1]))        
        self._mdt_exp.setText(str(df[7,1]))
    
    def outputTabUI(self):
        outputTabUI = qtWidget.QWidget()
        layout = qtWidget.QGridLayout()        
        
        self._sx = qtWidget.QCheckBox("Save files")        
        row = 0
        layout.addWidget(self._sx, row, 0)

        _mut_wave = qtWidget.QPushButton(self)
        _mut_wave.setText('Mutation wave plot')
        _mut_wave.clicked.connect(self.mutWaveAction)
        row = row + 1
        layout.addWidget(_mut_wave, row, 0)
        
        # _fit_wave = qtWidget.QPushButton(self)
        # _fit_wave.setText('Fitness wave plot')
        # _fit_wave.clicked.connect(self.fitWaveAction)
        # row = row + 1
        # layout.addWidget(_fit_wave, row, 0)
        
        _pop_growth = qtWidget.QPushButton(self)
        _pop_growth.setText('Population growth')
        _pop_growth.clicked.connect(self.popGrowthAction)
        row = row + 1
        layout.addWidget(_pop_growth, row, 0)
        
        _combined = qtWidget.QPushButton(self)
        _combined.setText('Combined mutation wave plot')
        _combined.clicked.connect(self.combinedMutWave)
        row = row + 1
        layout.addWidget(_combined, row, 0)
        
        _about = qtWidget.QPushButton(self)
        _about.setText('Short instruction')
        _about.clicked.connect(self.showAboutOutput)
        row = row + 1
        layout.addWidget(_about, row, 0)
        
        outputTabUI.setLayout(layout)
        return outputTabUI
    
    def showAboutOutput(self):
        self.showDialog(output_about, 'About')
        
    def showInactive(self):
        self.showDialog('Inactive', 'Info')
        
    def threadTabUI(self):
        threadTabUI = qtWidget.QWidget()
        layout = qtWidget.QFormLayout()
        
        _refresh = qtWidget.QPushButton("Refresh")
        _refresh.clicked.connect(self.update)
        
        layout.addRow("ID", _refresh)
        layout.addRow("Started", qtWidget.QLabel())
        
        for i in self.s_ID:
            layout.addRow(qtWidget.QLabel("ID: " + str(i)), qtWidget.QLabel())
        
        if self._th_pg.is_alive():
            layout.addRow(qtWidget.QLabel("population growth plot"), qtWidget.QLabel())
        if self._th_cp.is_alive():
            layout.addRow(qtWidget.QLabel("combined plot"), qtWidget.QLabel())
        if self._th_mw.is_alive():
            layout.addRow(qtWidget.QLabel("mutation wave"), qtWidget.QLabel())
        if self._th_fw.is_alive():
            layout.addRow(qtWidget.QLabel("fitness wave"), qtWidget.QLabel())
        
        threadTabUI.setLayout(layout)
        return threadTabUI
    
    def simStart(self):
        params, break_type = self.validateParams()
        
        try:
            if self._file_name == "" and self._sx.isChecked():
                raise Exception()
            if self._file_path == "" and self._sx.isChecked():
                raise Exception()
        except:
            self.showDialog("Enter save localization and name", "Alert")
            return
        
        plots = 16*self._sx.isChecked()
        
        iPop = []
        if self._analytical.isChecked():
            iPop = np.array([0,params[0]])
            iPop = sc.sparse.csr_matrix(iPop)
            select = 1
            name = 'analytical'
        else:
            iPop = np.ones(params[0])
            iMuts = np.zeros(params[0])
            iPop = sc.sparse.csr_matrix(np.array([iPop, iMuts]).T)
            select = 0
            name = 'normal'
          
        if self.ID == 1001:
            self.showDialog('To many simulations used. Please restart this app.', 'Memory alert')
            return 
        
        self.th_s.append(Process(target=seEvoInit, args=(copy.deepcopy(iPop), 
                                                    copy.deepcopy(params), 
                                                    self._file_name.text(), 
                                                    self._file_path.text(),
                                                    plots, self.q, self.ID, select, break_type))) 
        
        # seEvoInit(copy.deepcopy(iPop), 
        #     copy.deepcopy(params), 
        #     self._file_name.text(), 
        #     self._file_path.text(),
        #     plots, self.q, self.ID, select, break_type)
        
        self.s_ID.append(str(self.ID) + ', ' + name)
        self.ID = self.ID + 1
        self.th_s[self.idx_s].start()
        self.idx_s = self.idx_s + 1
        
    def simStop(self):
        msg = self._ID.text()
        try:
            msg = int(msg)
        except:
            self.showDialog('Type ID as integer in range (0,1000)', 'Error')
            return
        
        self.q.put(['1', str(msg), 'exit'])
    
    def validateParams(self):
        break_type = 0
        try:
            pop = int(self._population.text())
            cap = int(self._capacity.text())
            steps = self._steps.text().split(',')
            if len(steps) == 2:
                steps = int(steps[0])
                break_type = 1
            else:
                steps = int(steps[0])
            tau = float(self._tau.text())
            skip = float(self._skip.text())
            mut_prob = float(self._mut_prob.text())
            mut_effect = float(self._mut_effect.text())
            A = float(self._mdt_exp.text())
        except:
            self.showDialog("Type correct parameters","Alert")
            return
        return [pop, cap, steps, tau, skip, mut_prob, mut_effect, A], break_type
    
    def selectPath(self):
        dir_path = qtWidget.QFileDialog.getExistingDirectory(self,"Choose Directory","Z:/")
        self._file_path.setText(dir_path)

    def about(self):
        global disclaimer
        self.showDialog(disclaimer, 'About')

    def showDialog(self, text, title):
        msgBox = qtWidget.QMessageBox()
        msgBox.setIcon(qtWidget.QMessageBox.Information)
        msgBox.setText(text)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(qtWidget.QMessageBox.Ok)
         
        returnValue = msgBox.exec()

def run():
        app = qtWidget.QApplication(sys.argv)
        win = mainFormat()
        win.show()
        ret = app.exec_()
        sys.exit(ret)

if __name__ == "__main__":
    run()