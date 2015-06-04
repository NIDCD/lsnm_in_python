# ============================================================================
#
#                            PUBLIC DOMAIN NOTICE
#
#       National Institute on Deafness and Other Communication Disorders
#
# This software/database is a "United States Government Work" under the 
# terms of the United States Copyright Act. It was written as part of 
# the author's official duties as a United States Government employee and 
# thus cannot be copyrighted. This software/database is freely available 
# to the public for use. The NIDCD and the U.S. Government have not placed 
# any restriction on its use or reproduction. 
#
# Although all reasonable efforts have been taken to ensure the accuracy 
# and reliability of the software and data, the NIDCD and the U.S. Government 
# do not and cannot warrant the performance or results that may be obtained 
# by using this software or data. The NIDCD and the U.S. Government disclaim 
# all warranties, express or implied, including warranties of performance, 
# merchantability or fitness for any particular purpose.
#
# Please cite the author in any work or product based on this material.
# 
# ==========================================================================



# ***************************************************************************
#
#   Large-Scale Neural Modeling software (LSNM)
#
#   Section on Brain Imaging and Modeling
#   Voice, Speech and Language Branch
#   National Institute on Deafness and Other Communication Disorders
#   National Institutes of Health
#
#   This file (sim.py) was created on February 5, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on February 25 2015  
# **************************************************************************/

# sim.py
#
# Simulates delayed match-to-sample experiment using Wilson-Cowan neuronal
# population model.

# import regular expression modules (useful for reading weight files)
import re

# import random function modules
import random

# import math function modules
import math

# the following modules are imported from TVB library
from tvb.simulator.lab import *
import tvb.datatypes.time_series
from tvb.simulator.plot.tools import *
import tvb.simulator.plot.timeseries_interactive as ts_int
# end of TVB modules import

# import 'pyplot' modules to visualize outputs
import matplotlib.pyplot as plt

# import 'numpy' module, which contains useful matrix functions
import numpy as np

# import 'sys' module, which gives you access to file read/write functions
import sys

# import 'PyQt4' modules, which give you access to GUI functions
from PyQt4 import QtGui, QtCore

# create a class that will allow us to print output to our GUI widget
class MyStream(QtCore.QObject):

    message = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(MyStream, self).__init__(parent)

    def write(self, message):
        self.message.emit(str(message))

# create a class for our GUI and define its methods
class LSNM(QtGui.QWidget):

    def __init__(self):

        super(LSNM, self).__init__()
        
        self.initUI()

    def initUI(self):

        # the following three global variables are names of text files that contain
        # model definition, list of network weights, and experimental script to be
        # simulated
        model=''
        weights_list=''
        script=''

        # create a grid layout and set a spacing of 10 between widgets
        layout = QtGui.QGridLayout(self)
        layout.setSpacing(10)

        # Define what happens if users press EXIT on the toolbar
        exitAction = QtGui.QAction(QtGui.QIcon.fromTheme('exit'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        # create a push button object for opening file with model description
        uploadModelButton = QtGui.QPushButton('STEP ONE: Upload your model: ' + model, self)
        layout.addWidget(uploadModelButton, 0, 0)
        # define the action to be taken if upload model button is clicked on
        uploadModelButton.clicked.connect(self.browseModels)

        # create a text edit object for reading file with model description
        self.modelTextEdit = QtGui.QTextEdit()
        layout.addWidget(self.modelTextEdit, 1, 0)

        # create a push button for uploading file containing list of network weights
        uploadWeightsButton = QtGui.QPushButton('STEP TWO: Upload your weights: ' + weights_list, self)
        layout.addWidget(uploadWeightsButton, 0, 1)
        # define the action to be taken if upload weights button is clicked on
        uploadWeightsButton.clicked.connect(self.browseWeights)
        
        # create a text edit object for reading file with model description
        self.weightsTextEdit = QtGui.QTextEdit()
        layout.addWidget(self.weightsTextEdit, 1, 1)

        # create a button for uploading file containing experimental script to be simulated
        uploadScriptButton = QtGui.QPushButton('STEP THREE: Upload your script: ' + script, self)
        layout.addWidget(uploadScriptButton, 0, 2)
        # define the action to be taken if upload script button is clicked on
        uploadScriptButton.clicked.connect(self.browseScripts)

        # create a text edit object for reading file with model description
        self.scriptTextEdit = QtGui.QTextEdit()
        layout.addWidget(self.scriptTextEdit, 1, 2)

        # create a push button object labeled 'Run'
        runButton = QtGui.QPushButton('STEP FOUR: Run simulation', self)
        layout.addWidget(runButton, 0, 3)
        # define the action to be taken if Run button is clicked on
        runButton.clicked.connect(self.onStart)
    
        # define output display to keep user updated with simulation progress status
        self.runTextEdit = QtGui.QTextEdit()
        layout.addWidget(self.runTextEdit, 1, 3)

        # define progress bar to keep user informed of simulation progress status
        self.progressBar = QtGui.QProgressBar(self)
        self.progressBar.setRange(0,100)
        layout.addWidget(self.progressBar, 2, 3)
                        
        # create a push button object labeled 'Exit'
        exitButton = QtGui.QPushButton('Quit LSNM', self)
        layout.addWidget(exitButton, 2, 0)
        # define the action to be taken if Exit button is clicked on
        exitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

        # define the main thread as the main simulation code
        self.myLongTask = TaskThread()
        self.myLongTask.notifyProgress.connect(self.onProgress)
                
        # set the layout to the grid layout we defined in the lines above
        self.setLayout(layout)

        # set main window's size
        self.setGeometry(0, 0, 1300, 600)

        # set window's title
        self.setWindowTitle('Large-Scale Neural Modeling (LSNM)')
        
    def browseModels(self):

        global model
        # allow the user to browse files to find desired input file describing the modules
        # of the network
        model = QtGui.QFileDialog.getOpenFileName(self, 'Select *.txt file that contains model', '.')

        # open the file containing model description
        f = open(model, 'r')

        # display the contents of file containing model description
        with f:
            data = f.read()
            self.modelTextEdit.setText(data)
        
    def browseWeights(self):

        global weights_list
        # allow the user to browse files to find desired input file with a list of network weights
        weights_list = QtGui.QFileDialog.getOpenFileName(self, 'Select *.txt file that contains weights list', '.')

        # open file containing list of weights
        f = open(weights_list, 'r')

        # display contents of file containing list of weights
        with f:
            data = f.read()
            self.weightsTextEdit.setText(data)
        
    def browseScripts(self):

        global script
        # allow user to browse files to find desired input file containing experimental script
        # to be simulated
        script = QtGui.QFileDialog.getOpenFileName(self, 'Select *.txt file that contains script', '.')

        # open file containing experimental script
        f = open(script, 'r')

        # display contents of file containing experimental script
        with f:
            data = f.read()
            self.scriptTextEdit.setText(data)
        
    @QtCore.pyqtSlot()
    def onStart(self):
        self.myLongTask.start()

    def onProgress(self, i):
        self.progressBar.setValue(i)    
        
    def closeEvent(self, event):

        # display a message box to confirm user really intended to quit current session
        reply = QtGui.QMessageBox.question(self, 'Message',
                                           'Are you sure you want to quit LSNM?',
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
                                           QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    @QtCore.pyqtSlot(str)
    def on_myStream_message(self, message):
        self.runTextEdit.moveCursor(QtGui.QTextCursor.End)
        self.runTextEdit.insertPlainText(message)

            
class TaskThread(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self)

    notifyProgress = QtCore.pyqtSignal(int)
    
    def run(self):
            
        print 'Building network...'

        global noise

        # load a TVB simulation of a 74-node brain and uses it to provide variability
        # to an LSNM visual model network. It runs a simulation of the LSNM visual
        # network and writes out neural activities for each LSNM node and -relevant-
        # TVB nodes. Plots output as well.

        ########## THE FOLLOWING SIMULATES TVB NETWORK'S #######################
        # The TVB Wilson Cowan simulation has been preprocessed and it is located
        # in an 'npy' data file. Thus, erase the commments from the following
        # 'np.load' if you just need to load that data file onto a numpy array
        # The data file contains an array of 5 dimensions as follows:
        # [timestep, state_variable_E, state_variable_I, node_number, mode]
        #RawData = np.load("wilson_cowan_brain_74_nodes.npy")

        # define white matter transmission speed in mm/ms for TVB simulation
        TVB_speed = 4.0

        # define length of TVB simulation in ms
        simulation_length = 6500

        # define global coupling strength as in Sanz-Leon et al (2015), figure 17,
        # 3rd column, 3rd row
        TVB_global_coupling_strength = 0.0042

        # define the population model to be used and state variables to be collected.
        # the parameters below were taken from in Sanz-Leon et al (2015), table 11,
        # case 'a' 
        TVB_WC = models.WilsonCowan(variables_of_interest=['E','I'],
                                    r_i=1, r_e=1, k_e=1, k_i=1, tau_e=10, tau_i=10,
                                    c_ee=12, c_ei=4, c_ie=13, c_ii=11, alpha_e=1, alpha_i=1,
                                    a_e=1.2, a_i=1, b_e=2.8, b_i=4, c_e=1, c_i=1,
                                    P=0, Q=0)

        # now load white matter connectivity (74 ROI matrix from TVB demo set)
        white_matter = connectivity.Connectivity(load_default=True)
        #white_matter.configure()

        # Define the transmission speed of white matter tracts (4 mm/ms)
        white_matter.speed = numpy.array([TVB_speed])

        # Define the coupling function between white matter tracts and brain regions
        white_matter_coupling = coupling.Linear(a=TVB_global_coupling_strength)

        #Initialise an Integrator
        # set numpy's seed
        my_seed = 13
        my_random_state = numpy.random.RandomState(my_seed)
        hiss = noise.Additive(nsig=0.08)
        heunint = integrators.EulerStochastic(dt=2 ** -2, noise=hiss)
        heunint.configure()

        # Define a monitor to be used for TVB simulation (i.e., which simulated data is
        # going to be collected
        #what_to_watch = monitors.Raw(variables_of_interest=['E','I'])
        what_to_watch = monitors.SubSample(period=5.0)
        
        # Initialize a TVB simulator
        TVB_sim = simulator.Simulator(model=TVB_WC, connectivity=white_matter,
                                      coupling=white_matter_coupling,
                                      integrator=heunint, monitors=what_to_watch)

        TVB_sim.configure()

        # sample TVB raw data array to extract 220 data points (for plotting only)
        #RAW = RawData[::400]    # round(88000 / 220) = 400

        # define the simulation time in total number of timesteps
        # Each timestep is roughly equivalent to 5ms
        # (each trial is 4400 timesteps long x 12 trials = 52800)
        LSNM_simulation_time = 1300
        
        # sample TVB raw data array file to extract 1100 data points
        # (only use if you are loading a preprocessed TVB simulation)
        #TVB_sampling_rate = int(round(88000 / simulation_time))
        #RawData = RawData[::TVB_sampling_rate]

        # To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
        # we are assuming that each simulation timestep is equivalent to 5 milliseconds
        # of real time. 
        
        # print RawData.shape
        
        # The TVB brain areas where our LSNM units are going to be embedded it
        # hardcoded for now, but will be included in as an option in the LSNM GUI.
        
        # LSNM V1 module is embedded into TVB rV1 (node 72 in TVB)
        # LSNM A1 module is embedded into TVB rA1 (node 37 in TVB)
        # LSNM IT module is embedded into TVB rTCi (node 68 in TVB, i.e., inferior temporal cortex)
        # LSNM STG module is embedded into TVB rTCs (node 70 in TVB, i.e., superior temporal cortex)
        # LSNM FS module is embedded into TVB rPFCvl (node 60 in TVB, i.e., ventrolateral prefrontal cortex)
        # create a dictionary linking LSNM modules and TVB nodes
        lsnm_tvb_link = {'ev1v': 72,
                         'ev1h': 72,
                         'ea1u': 37,
                         'ea1d': 37,
                         'exss': 68,
                         'estg': 70,
                         'exfs': 60 }

        # declare a gain for the link from TVB to LSNM
        lsnm_tvb_gain = 0.001

        # declare a integration interval for the 'integrated' synaptic activity,
        # for fMRI computation, in number of timesteps.
        # The same variable is used to know how often we are going to write to
        # output files
        synaptic_interval = 10
                
        # print which brain areas from TVB we are using,
        # as well as 'first degree' connections of the TVB areas listed
        # the folowing printout is only for informational purposes

        print '\rIncoming units from TVB are: '

        print '\rInto ' + white_matter.region_labels[72],
        print ': ',
        print white_matter.region_labels[np.nonzero(white_matter.weights[72])]
        print 'with the following weights: ',
        print white_matter.weights[72][np.nonzero(white_matter.weights[72])]
        
        print '\rInto ' + white_matter.region_labels[68],
        print ': ',
        print white_matter.region_labels[np.nonzero(white_matter.weights[68])]
        print 'with the following weights: ',
        print white_matter.weights[68][np.nonzero(white_matter.weights[68])]
        
        print '\rInto ' + white_matter.region_labels[60],
        print ': ',        
        print white_matter.region_labels[np.nonzero(white_matter.weights[60])]
        print 'with the following weights: ',
        print white_matter.weights[60][np.nonzero(white_matter.weights[60])]
        
        print '\rInto ' + white_matter.region_labels[37],
        print ': ',
        print white_matter.region_labels[np.nonzero(white_matter.weights[37])]
        print 'with the following weights: ',
        print white_matter.weights[37][np.nonzero(white_matter.weights[37])]
        
        print '\rInto ' + white_matter.region_labels[70],
        print ': ',
        print white_matter.region_labels[np.nonzero(white_matter.weights[70])]
        print 'with the following weights: ',
        print white_matter.weights[70][np.nonzero(white_matter.weights[70])]
        

        ######### THE FOLLOWING SIMULATES LSNM NETWORK ########################
        # initialize an empty list to store ALL of the modules of the neural network
        modules = []

        # open the input file containing module declarations (i.e., the 'model'), then
        # load the file into a python list of lists and close file safely
        f = open(model, 'r')
        try: 
            modules = [line.split() for line in f]
        finally:
            f.close()
    
        # convert ALL module dimensions to integers since we will need those numbers
        # later
        for module in modules:
            module[1] = int(module[1])
            module[2] = int(module[2])
    
        # convert ALL parameters in the modules to float since we will need to use those
        # to solve Wilson-Cowan equations
        for module in modules:
            module[4] = float(module[4])
            module[5] = float(module[5])
            module[6] = float(module[6])
            module[7] = float(module[7])
            module[8] = float(module[8])
            module[9] = float(module[9])

        # add a list of units to each module, using the module dimensions specified
        # in the input file (x_dim * y_dim) and initialize all units in each module to 'initial_value'
        # It also adds three extra elements per each unit to store (1) sum of all incoming activity,
        # (2) sum of inbititory, and (3) sum
        # of excitatory activity, at the current time step. It also add an empty list, '[]', to store
        # list of outgoing weights
        for module in modules:
            # remove initial value from the list
            initial_value = module.pop()
            x_dim = module[1]
            y_dim = module[2]
    
            # create a matrix for each unit in the module, to contain unit value,
            # total sum of inputs, sum of excitatory inputs, sum of inhibitory inputs,
            # and connection weights
            unit_matrix = [[[initial_value, 0.0, 0.0, 0.0, []] for x in range(y_dim)] for y in range(x_dim)]

            # now append that matrix to the current module
            module.append(unit_matrix)

        # now turn the list modules into a dictionary so we can access each module using the
        # module name as key (this makes index 0 dissapear and shifts all other list indexes by 1)
        modules = {m[0]: m[1:] for m in modules}

        # read file that contains list of weight files, store the list of files in a python list,
        # and close the file safely
        f = open(weights_list, 'r')
        try:  
            weight_files = [line.strip() for line in f]
        finally:
            f.close()

        # build a dictionary of replacements for parsing the weight files
        replacements = {'Connect': '',
                        'From:': '',
                        '(':'[',
                        ')':']',
                        '{':'[',
                        '}':']',
                        '|':''}

        # the following variable counts the total number of synapses in the network (for
        # informational purposes
        synapse_count = 0
    
        # open each weight file in the list of weight files, one by one, and transfer weights
        # from those files to each unit in the module list
        # Note: file f is closed automatically at the end of 'with' since block 'with' is a
        # context manager for file I/O
        for file in weight_files:
            with open(file) as f:

                # read the whole file and store it in a string
                whole_thing = f.read()
        
                # find which module is connected to which module
                module_connection = re.search(r'Connect\((.+?),(.+?)\)', whole_thing)

                # get rid of white spaces from origin and destination modules
                origin_module = module_connection.group(1).strip()
                destination_module = module_connection.group(2).strip()

                # gets rid of C-style comments at the beginning of weight files
                whole_thing = re.sub(re.compile('%.*?\n'), '', whole_thing)

                # removes all white spaces (space, tab, newline, etc) from weight files
                whole_thing = ''.join(whole_thing.split())
        
                # replaces Malle-style language with python lists characters
                for i, j in replacements.iteritems():
                    whole_thing = whole_thing.replace(i, j)

                # now add commas between pairs of brackets
                whole_thing = whole_thing.replace('][', '],[')

                # now insert commas between right brackets and numbers (temporary hack!)
                whole_thing = whole_thing.replace(']0', '],0')
                whole_thing = whole_thing.replace(']1', '],1')
                whole_thing = whole_thing.replace(']-', '],-')

                # add extra string delimiters to origin_module and destination_module so
                # that they can be recognized as python "strings" when the list or lists
                # is formed
                whole_thing = whole_thing.replace(origin_module+','+destination_module,
                                                  "'"+origin_module+"','"+destination_module+"'", 1)

                # now convert the whole thing into a python list of lists, using Python's
                # own interpreter 
                whole_thing = eval(whole_thing)

                # remove [origin_module, destination_module] from list of connections
                whole_thing = whole_thing[1]
        
                # now groups items in the form: [(origin_unit), [[[destination_unit], weight],
                # ..., [[destination_unit_2], weight_2]])]
                whole_thing = zip(whole_thing, whole_thing[1:])[::2]
                
                # insert [destination_module, x_dest, y_dest, weight] in the corresponding origin
                # unit location of the modules list while adjusting (x_dest, y_dest) coordinates
                # to a zero-based format (as used in Python)
                for connection in whole_thing:
                    for destination in connection[1]:
                        modules[origin_module][8][connection[0][0]-1][connection[0][1]-1][4].append (
                            [destination_module,        # insert name of destination module
                            destination[0][0]-1,         # insert x coordinate of destination unit
                            destination[0][1]-1,         # insert y coordinate of destination unit
                            destination[1]])           # insert connection weight
                        synapse_count += 1

        # the following stores values over time of all units (electrical activity, synaptic activity,
        # and tvb activity) to output data files in text format

        fs_neuronal = []
        fs_synaptic = []
        fs_tvb      = []
        

        # open one output file per module to record electrical and synaptic activities
        for module in modules.keys():
            # open one output file per module
            fs_neuronal.append(open('./output/' + module + '.out', 'w'))
            fs_synaptic.append(open('./output/' + module + '_synaptic.out', 'w'))

        # ...for TVB node activities, open one output file per 'host' node to record timeseries
        # TVB 'host' nodes
        for host_tvb_node in lsnm_tvb_link.keys():
            fs_tvb.append(open('./output/' + host_tvb_node + '_tvb.out', 'w'))
        
    
        # create a dictionary so that each module name is associated with one output file
        fs_dict_neuronal = dict(zip(modules.keys(), fs_neuronal))
        fs_dict_synaptic = dict(zip(modules.keys(), fs_synaptic))

        # also create a dictionary for host TVB node activity output files
        fs_dict_tvb = dict(zip(lsnm_tvb_link.keys(), fs_tvb)) 
        
        # open the file with the experimental script and store the script in a string
        with open(script) as s:
            experiment_script = s.read()
            
        # initialize number of timesteps for simulation
        sim_percentage = 100.0/LSNM_simulation_time

        # run the simulation for the number of timesteps given
        print '\r Running simulation...'

        # uncomment the following line and subsititute in the for loop below if you want
        # LSNM to drive the whole simulation
        #for t in range(LSNM_simulation_time):

        # initialize timestep counter for LSNM timesteps
        t = 0

        # the following 'for loop' is the main loop of the TVB simulation with the parameters
        # defined above. Note that the LSNM simulator is literally embedded into the TVB
        # simulation and both run concurrently, timestep by timestep.
        for raw in TVB_sim(simulation_length=simulation_length):

            # let the user know the percentage of simulation that has elapsed
            self.notifyProgress.emit(int(round(t*sim_percentage,0)))
            
            # run the experimental script given by user's script file
            exec(experiment_script)
                    
            # The following 'for loop' computes sum of excitatory and sum of inhibitory activities
            # at destination nodes using destination units and connecting weights provided
            for m in modules.keys():
                for x in range(modules[m][0]):
                    for y in range(modules[m][1]):
                
                        # we are going to do the following only for those units in the network that
                        # have weights that project to other units elsewhere

                        # extract value of origin unit (unit projecting weights elsewhere)
                        origin_unit = modules[m][8][x][y][0]
                
                        for w in modules[m][8][x][y][4]:
                        
                            # First, find outgoing weights for all destination units and (except
                            # for those that do not
                            # have outgoing weights, in which case do nothing) compute weight * value
                            # at destination units
                            dest_module = w[0]
                            x_dest = w[1]
                            y_dest = w[2]
                            weight = w[3]
                            value_x_weight = origin_unit * weight 
                        
                            # Now, accumulate (i.e., 'integrate') & store those values at the
                            # destination units data structure,
                            # to be used later during neural activity computation.
                            # Note: Keep track of inhibitory and excitatory input summation
                            # separately, as shown below:
                            if value_x_weight > 0:
                                modules[dest_module][8][x_dest][y_dest][2] += value_x_weight
                            else:
                                modules[dest_module][8][x_dest][y_dest][3] += value_x_weight

                            # ... but also keep track of the total input summation, as shown
                            # below. The reason for this is that we need the input summation
                            # to each neuronal population unit AT EACH TIME STEP, as well as
                            # the excitatory and inhibitory input summations accumulated OVER
                            # A NUMBER OF TIMESTEPS (that number is usually 10). We call such
                            # accumulation of inputs
                            # over a number of timesteps the 'integrated synaptic activity'
                            # and it is used to compute fMRI and MEG.
                            modules[dest_module][8][x_dest][y_dest][1] += value_x_weight

            # the following 'for loop' goes through each LSNM module that is 'embedded' into The Virtual
            # Brain, and adds the product of each TVB -> LSNM unit value times their respective
            # connection weight (provided by white matter tract weights) to the sum of excitatory
            # activities of each embedded LSNM unit. THIS IS THE STEP
            # WHERE THE INTERACTION BETWEEN LSNM AND TVB HAPPENS. THAT INTERACTION IS SO FAR
            # UNIDIRECTIONAL TVB -> LSNM, but will add a feedback connection soon.
            # Please note that whereas the previous 'for loop' goes though the network updating
            # unit sum of activities at destination units, the 'for loop' below goes through the
            # network updating the sum of activities of the CURRENT unit

            # we are going to do the following only for those modules/units in the LSNM
            # network that have connections from TVB nodes
            for m in lsnm_tvb_link.keys():
                if modules.has_key(m):

                    # extract TVB node number where module is embedded
                    tvb_node = lsnm_tvb_link[m]

                    # extract TVB node numbers that are conected to TVB node above
                    tvb_conn = np.nonzero(white_matter.weights[tvb_node])
                    # extract the numpy array from it
                    tvb_conn = tvb_conn[0]

                    # build a numpy array of weights from TVB connections to TVB homologous nodes
                    wm = white_matter.weights[tvb_node][tvb_conn]
                    
                    # now go through all the units of current LSNM modules...
                    for x in range(modules[m][0]):
                        for y in range(modules[m][1]):

                            # do the following for each white matter connection to current LSNM unit
                            for i in range(tvb_conn.size):
                                
                                # extract the value of TVB node from preprocessed raw time series
                                # uncomment if you want to use preprocessed TVB timeseries
                                #value =  RawData[t, 0, tvb_conn[i]]

                                # convert current TVB electrical activity to a numpy array 
                                RawData = numpy.array(raw[0][1])

                                # extract value of TVB node
                                value = RawData[0, tvb_conn[i]]
                                value =  value[0]
                                
                                # calculate a incoming weight by applying a gain into the LSNM unit
                                weight = wm[i] * lsnm_tvb_gain
                                value_x_weight = value * weight
                        
                                # ... and add the incoming value_x_weight to the summed synaptic
                                # activity of the current unit
                                if value_x_weight > 0:
                                    modules[m][8][x][y][2] += value_x_weight
                                else:
                                    modules[m][8][x][y][3] += value_x_weight
                                # ... but store the total of inputs separately as well
                                modules[m][8][x][y][1] += value_x_weight

            # the following variable will keep track of total number of units in the network
            unit_count = 0


            # write the neural and synaptic activity to output files of each unit at a given
            # timestep interval, given by the variable <synaptic interval>.
            # The reason we write to the output files before we do any computations is that we
            # want to keep track of the initial values of each unit in all modules
            for m in modules.keys():
                for x in range(modules[m][0]):
                    for y in range(modules[m][1]):
                        
                        # Write out neural and integrated synaptic activity, and reset
                        # integrated synaptic activity, but ONLY IF a given number of timesteps
                        # has elapsed (integration interval)
                        if ((LSNM_simulation_time + t) % synaptic_interval) == 0:
                            # write out neural activity first...
                            fs_dict_neuronal[m].write(repr(modules[m][8][x][y][0]) + ' ')
                            # now calculate and write out synaptic activity...
                            synaptic = modules[m][8][x][y][2] + abs(modules[m][8][x][y][3])
                            fs_dict_synaptic[m].write(repr(synaptic) + ' ')
                            # ...finally, reset synaptic activity (but not neural activity).
                            modules[m][8][x][y][2] = 0.0
                            modules[m][8][x][y][3] = 0.0                            

                        
                # finally, insert a newline character so we can start next set of units on a
                # new line
                fs_dict_neuronal[m].write('\n')
                fs_dict_synaptic[m].write('\n')

            # also write neural activity of TVB host nodes to output files at the current
            # time step
            for host_node in lsnm_tvb_link.keys():
                host_node_value = RawData[0, lsnm_tvb_link[host_node]]
                host_node_value = host_node_value[0]
                fs_dict_tvb[host_node].write(repr(host_node_value) + ' ')
                            

            
            # the following 'for loop' computes the neural activity at each unit in the network,
            # depending on their 'activation rule'
            for m in modules.keys():
                for x in range(modules[m][0]):
                    for y in range(modules[m][1]):
                        # if the current module is an LSNM unit, use in-house wilson-cowan
                        # algorithm below (based on original Tagamets and Horwitz, 1995)
                        if modules[m][2] == 'wilson_cowan':
                        
                            # extract Wilson-Cowan parameters from the list
                            threshold = modules[m][3]
                            noise = modules[m][7]
                            K = modules[m][6]
                            decay = modules[m][5]
                            Delta = modules[m][4] 

                            # compute input to current unit
                            in_value = modules[m][8][x][y][1]

                            # now subtract the threshold parameter from that sum
                            in_value = in_value - threshold

                            # now compute a random value between -0.5 and 0.5
                            r_value = random.uniform(0,1) - 0.5

                            # multiply it by the noise parameter and add it to input value
                            in_value = in_value + r_value * noise

                            # now multiply by parameter K and apply sigmoid function e
                            sigmoid = 1.0 / (1.0 + math.exp(-K * in_value))
                        
                            # now multiply sigmoid by delta parameter, subtract decay parameter,
                            # ... and add all to current value of unit (x, y) in module m
                            modules[m][8][x][y][0] += Delta * sigmoid - decay * modules[m][8][x][y][0]

                            # now reset the sum of excitatory and inhibitory weigths at each unit,
                            # since we only need it for the current timestep (new sums of excitatory and
                            # inhibitory unit activations will be computed at the next time step)
                            modules[m][8][x][y][1] = 0.0
                            
                        unit_count += 1
                        
            # increase the number of timesteps
            t = t + 1
                        
        # be safe and close output files properly
        for f in fs_neuronal:
            f.close()
        for f in fs_synaptic:
            f.close()
        for f in fs_tvb:
            f.close()

        print '\r Simulation Finished.'
        print '\r Output data files saved.'

        
def main():
    
    # create application object called 'app'
    app = QtGui.QApplication([])

    # create a widget window called "lsnm"
    lsnm = LSNM()

    lsnm.show()

    myStream = MyStream()
    myStream.message.connect(lsnm.on_myStream_message)

    sys.stdout = myStream
    
    # main loop of application with a clean exit
    sys.exit(app.exec_())
        
# the following is the standard boilerplate that calls the main() function
if __name__ == '__main__':
    main()


