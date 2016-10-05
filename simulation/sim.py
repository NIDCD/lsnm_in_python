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
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on April 16 2016
#
#   Based on computer code originally developed by Malle Tagamets and
#   Barry Horwitz (Tagamets and Horwitz, 1998)
# **************************************************************************/

# sim.py
#
# Simulates delayed match-to-sample experiment using Wilson-Cowan neuronal
# population model.

# import regular expression modules (useful for reading weight files)
import re

# import random function modules
import random as rdm

# import math function modules
import math

# import json module for storing output data files
import json

# import time module (needed for recording start and end simulation times in log file
import time as time_module

try:
    # only import TVB modules if simulation requires TVB connectome
    # the following modules are imported from TVB library
    from tvb.simulator.lab import *
    import tvb.datatypes.time_series
    from tvb.simulator.plot.tools import *
    import tvb.simulator.plot.timeseries_interactive as ts_int
    # end of TVB modules import
except ImportError:
    pass

# import 'pyplot' modules to visualize outputs
import matplotlib.pyplot as plt

# import 'numpy' module, which contains useful matrix functions
import numpy as np

# module to import external or user created modules into current script
import importlib as il

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

        global learning
        learning = False                 # Determines whether hebbian learning will be applied

        global learning_file             # name of the file that contains name of modules that will undergo
        learning_file =""                # hebbian learning

        global useTVBConnectome
        useTVBConnectome = False         # Determines whether to use a TVB connectome within simulation

        global embedding_file            # locations of embedded LSNM nodes within TVB connectome
        embedding_file = ""

        global generateSubject
        generateSubject = False         # Determines whether or not to vary connection weights given
                                         # to create a new subject for the current simulation

        global model                     # contains name of file describing module and parameters
        model = ""
        
        global weights_list              # contains name of file with list of connection weights
        weights_list = ""
        
        global script                    # contains name of file with simulation script
        script = ""
        
        global neural_net                # contains name of file with neural network
        neural_net = ""
        
        # create a grid layout and set a spacing of 10 between widgets
        layout = QtGui.QGridLayout(self)
        layout.setSpacing(10)

        # Define what happens if users press EXIT on the toolbar
        exitAction = QtGui.QAction(QtGui.QIcon.fromTheme('exit'), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        # create a push button object for opening file with model description
        uploadModelButton = QtGui.QPushButton('Model description file: ' + model, self)
        layout.addWidget(uploadModelButton, 0, 0)
        # define the action to be taken if upload model button is clicked on
        uploadModelButton.clicked.connect(self.browseModels)

        # create a push button for uploading file containing list of network weights
        uploadWeightsButton = QtGui.QPushButton('Weights list file: ' + weights_list, self)
        layout.addWidget(uploadWeightsButton, 1, 0)
        # define the action to be taken if upload weights button is clicked on
        uploadWeightsButton.clicked.connect(self.browseWeights)
        
        # create a button for uploading file containing experimental script to be simulated
        uploadScriptButton = QtGui.QPushButton('Simulation script file: ' + script, self)
        layout.addWidget(uploadScriptButton, 2, 0)
        # define the action to be taken if upload script button is clicked on
        uploadScriptButton.clicked.connect(self.browseScripts)

        # create a push button object labeled 'Neural Net'
        neuralNetButton = QtGui.QPushButton('File containing neural net', self)
        layout.addWidget(neuralNetButton, 3, 0)
        # define the action to be taken if Run button is clicked on
        neuralNetButton.clicked.connect(self.browseNeuralNets)

        # define check box to allow user to define LSNM module that will participate as a source
        # in hebbian learning
        learningBox = QtGui.QCheckBox('Use Hebbian Learning', self)
        layout.addWidget(learningBox, 4, 0)
        # define the action to be taken if learning box is selected
        learningBox.stateChanged.connect(self.learningOrNot)

        # define checkbox to allow users to determine whether or not a TVB connectome will be used
        # in simulation
        connectomeBox = QtGui.QCheckBox('Use TVB Connectome', self)
        layout.addWidget(connectomeBox, 5, 0)
        connectomeBox.stateChanged.connect(self.connectomeOrNot)

        # create a push button object labeled 'Run'
        runButton = QtGui.QPushButton('Run simulation', self)
        layout.addWidget(runButton, 7, 0)
        # define the action to be taken if Run button is clicked on
        runButton.clicked.connect(self.onStart)
    
        # create a push button object labeled 'Exit'
        exitButton = QtGui.QPushButton('Quit LSNM', self)
        layout.addWidget(exitButton, 8, 0)
        # define the action to be taken if Exit button is clicked on
        exitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)

        # create a text edit object for reading file with model description
        self.modelTextEdit = QtGui.QLineEdit()
        layout.addWidget(self.modelTextEdit, 0, 1)

        # create a text edit object for reading file with model description
        self.weightsTextEdit = QtGui.QLineEdit()
        layout.addWidget(self.weightsTextEdit, 1, 1)

        # create a text edit object for reading file with experiment script
        self.scriptTextEdit = QtGui.QLineEdit()
        layout.addWidget(self.scriptTextEdit, 2, 1)

        # create a text edit object for reading file with both model and weights (neural net)
        self.neuralNetTextEdit = QtGui.QLineEdit()
        layout.addWidget(self.neuralNetTextEdit, 3, 1)

        # create a text edit object for reading file that contains modules that will undergo learning
        self.learningTextEdit = QtGui.QLineEdit()
        layout.addWidget(self.learningTextEdit, 4, 1)

        # create text edit object that contains name of file with locations of LSNM nodes within TVB
        self.embeddingTextEdit = QtGui.QLineEdit()
        layout.addWidget(self.embeddingTextEdit, 5, 1)

        # define checkbox to allow users to determine whether or not to vary weights randomly from
        # weights given to create a new simulated subject
        newSubjectBox= QtGui.QCheckBox('Vary weights to create new subject', self)
        layout.addWidget(newSubjectBox, 6, 0)
        newSubjectBox.stateChanged.connect(self.createNewSubject)
        
        # define output display to keep users updated with simulation progress status messages
        self.runTextEdit = QtGui.QTextEdit()
        layout.addWidget(self.runTextEdit, 7, 1)

        # define progress bar to keep user informed of simulation progress status
        self.progressBar = QtGui.QProgressBar(self)
        self.progressBar.setRange(0,100)
        layout.addWidget(self.progressBar, 8, 1)
                        
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

        # display the name of file containing model description
        self.modelTextEdit.setText(model)
        
    def browseWeights(self):

        global weights_list
        # allow the user to browse files to find desired input file with a list of network weights
        weights_list = QtGui.QFileDialog.getOpenFileName(self, 'Select *.txt file that contains weights list', '.')

        # display name of file containing list of weights
        self.weightsTextEdit.setText(weights_list)
        
    def browseScripts(self):

        global script
        # allow user to browse files to find desired input file containing experimental script
        # to be simulated
        script = QtGui.QFileDialog.getOpenFileName(self, 'Select *.txt file that contains script', '.')

        # display contents of file containing experimental script
        self.scriptTextEdit.setText(script)

    def browseNeuralNets(self):

        global neural_net
        # allow user to browse files to find desired input file containing neural net (model and weights
        # combined)
        neural_net = QtGui.QFileDialog.getOpenFileName(self, 'Select *.json file that contains neural net', '.')

        # display contents of file containing neural net (model and weights combined)
        self.neuralNetTextEdit.setText(neural_net)

    def learningOrNot(self, state):

        global learning
        global learning_file
        # allow user to decide whether network weights will be participating in Hebbian learning during
        # simulation
        if state == QtCore.Qt.Checked:
            learning = True
            print '\rUsing Hebbian learning...'
            learning_file = QtGui.QFileDialog.getOpenFileName(self, 'Select file that contains learning modules', '.')

            # display contents of file with names of modules whose weights will undergo hebbian learning
            self.learningTextEdit.setText(learning_file)
        else:
            learning = False
            print '\rNot Using learning...'
        
    def connectomeOrNot(self, state):

        global useTVBConnectome
        global embedding_file
        # allow user to decide whether to use a TVB connectome as part of the simulation
        if state == QtCore.Qt.Checked:
            useTVBConnectome = True
            print '\rUsing TVB Connectome...'
            embedding_file = QtGui.QFileDialog.getOpenFileName(self,'Select file that contains locations of LSNM nodes', '.')

            # display contents of file with locations of LSNM nodes with TVB connectome
            self.embeddingTextEdit.setText(embedding_file)
        else:
            useTVBConnectome = False
            print '\rNOT Using TVB Connectome...'

    def createNewSubject(self, state):

        global generateSubject
        # allow user to decide whether or not to vary weights given to generate new subject
        if state == QtCore.Qt.Checked:
            generateSubject = True
            print '\rGenerating new subject by randomly varying connection weights given...'
        else:
            generateSubject = False
            print '\rWe are NOT generating a new subject...'
            
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

class WilsonCowanPositive(models.WilsonCowan):
    "Declares a class of Wilson-Cowan models that use the default TVB parameters but"
    "only allows values between 0 and 1 at integration time. In other words, it clips state"
    "variables to the range [0,1] when a stochastic integration is used"
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        state_variables[state_variables < 0.0] = 0.0
        state_variables[state_variables > 1.0] = 1.0
        return super(WilsonCowanPositive, self).dfun(state_variables, coupling, local_coupling)
            
class TaskThread(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self)

    notifyProgress = QtCore.pyqtSignal(int)
    
    def run(self):

        start_time = time_module.asctime(time_module.localtime(time_module.time()))
        print '\rStart Time: ', start_time 
        print '\rBuilding network...'

        global noise

        # load a TVB simulation of a 998-node brain and uses it to provide variability
        # to an LSNM visual model network. It runs a simulation of the LSNM visual
        # network and writes out neural activities for each LSNM node and -relevant-
        # TVB nodes. Plots output as well.

        # define a flag that tells the network whether to send feedback connections
        # from LSNM to TVB
        FEEDBACK = True

        # define a number that the simulator will use to generate new subjects. If this
        # option is checked at simulation time, the simulator with multiply the connection
        # weights given by a random amount of between the number given and 1.0
        subject_variation = 0.98

        # define how many milliseconds there are in each simulation timestep
        timesteps_to_time_constant = 5.0
        
        # define white matter transmission speed in mm/ms for TVB simulation
        TVB_speed = 4.0

        # define global coupling strength as in Sanz-Leon et al (2015), figure 17,
        # 3rd column, 3rd row
        TVB_global_coupling_strength = 0.0042

        # declare a variable that describes number of nodes in TVB connectome
        TVB_number_of_nodes = 998
        
        # declare a file name for the output file where the neural network structure will be
        # stored (modules and weights among modules)
        neural_network = 'neuralnet.json'

        # declare a file name for the output file that contains simulation log
        log_file = 'log.txt'

        # the following are the weights used among excitatory and inhibitory populations
        # in the TVB's implementation of the Wilson-Cowan equations. These values were
        # taken from the default values in the TVB source code in script "models.npy"
        # The values are also in Table 11 Column a of Sanz-Leon et al (2015). Note that
        # the subscripts in Sanz-Leon's notation are reversed, thus w_ei, in her notation,
        # means a weight from inhibitory to excitatory. In our notation, W_ei means a
        # weight from excitatory to inhibitory.
        #
        # The reason we re-defined the weights below is to be able to calculate local
        # synaptic activity at each timestep, as the TVB original source code does not
        # calculate synaptic activities
        w_ee = 12.0
        w_ii = 11.0
        w_ei = 13.0
        w_ie =  4.0
        
        # now load white matter connectivity (998 ROI matrix from TVB demo set, AKA Hagmann's connectome)
        if useTVBConnectome == True:
            white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
        
            # Define the transmission speed of white matter tracts (4 mm/ms)
            white_matter.speed = numpy.array([TVB_speed])

            # Define the coupling function between white matter tracts and brain regions
            white_matter_coupling = coupling.Linear(a=TVB_global_coupling_strength)

            #Initialize an Integrator for TVB
            euler_int = integrators.EulerStochastic(dt=5, noise=noise.Additive(nsig=0.01))
            euler_int.configure()

            # Define a monitor to be used for TVB simulation (i.e., which simulated data is
            # going to be collected
            what_to_watch = monitors.Raw()
        
            # Initialize a TVB simulator
            sim = simulator.Simulator(model=WilsonCowanPositive(), connectivity=white_matter,
                                      coupling=white_matter_coupling,
                                      integrator=euler_int, monitors=what_to_watch)

            sim.configure()

            # To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
            # we are assuming that each simulation timestep is equivalent to 5 milliseconds
            # of real time. 
        
            # The TVB brain areas where our LSNM units are going to be embedded are
            # contained in a file called "lsnm_tvb_link.txt"
        
            # create a python dictionary of LSNM modules and the location of the corresponding
            # TVB node in which the TVB node is to be embedded. In other words, the closest TVB node
            # is  used as a 'host' node to embed a given LSNM module

            lsnm_tvb_link = {}

            # open the input file containing LSNM <-> TVB link declarations, then
            # load the file into a python list of lists and close file safely
            with open(embedding_file) as f:
                for line in f:
                    (key, val) = line.split()
                    lsnm_tvb_link[key] = int(val)

            print '\r LSNM <-> TVB Link: ', lsnm_tvb_link
            
            # create two arrays to store synaptic activity for each and all TVB nodes,
            # one for absolute values of synaptic activities (for fMRI computation):
            tvb_abs_syna = []
            # ... and the other one for signed values of synaptic activity (for MEG computation):
            tvb_signed_syna = []
            # also, create an array to store electrical activity for all TVB nodes
            tvb_elec = []

            # create and initialize array to store synaptic activity for all TVB nodes, excitatory
            # and inhibitory parts.
            # The synaptic activity for each node is zero at first, then it accumulates values
            # (integration) during a given number of timesteps. Every number of timesteps
            # (given by 'synaptic_interval'), the array below is re-initialized to zero.
            # One array is for accumulating the absolute value of synaptic activity (for BOLD computation):
            current_tvb_abs_syn =    [ [0.0]*TVB_number_of_nodes for _ in range(2) ]
            # ... and another array for accumulating the signed values of synaptic activity (for MEG computation):
            current_tvb_signed_syn = [ [0.0]*TVB_number_of_nodes for _ in range(2) ]
        
        # number of units in each LSNM sub-module
        n = 81
        
        # declare a gain for the link from TVB to LSNM (around which normally distributed
        # random numbers will be generated). I obtained this number of diving the TVB gain used
        # within the connectome nodes by 81 (# of units in each LSNM module).
        # Please note that each module can have a different number of 9x9 or 1x81 matrices, and
        # so that needs to be taken into account as well by multiplying the number of units in
        # a matrix times the number of modules contained in an embedded brain region.
        lsnm_tvb_gain = TVB_global_coupling_strength / n
        
        # declare an integration interval for the 'integrated' synaptic activity,
        # for fMRI computation, in number of timesteps.
        # The same variable is used to know how often we are going to write to
        # output files
        synaptic_interval = 10
                   
        ######### THE FOLLOWING SIMULATES LSNM NETWORK ########################
        # initialize an empty list to store ALL of the modules of the LSNM neural network
        # NOTE: This is the main data structure holding all of the LSNM network values
        # at each timestep, including neural activity, connections weights, neural
        # population model parameters, synaptic activity, module dimensions, among others.

        # if no model and weights list were specified, upload the whole network (modules and weights)
        # from neural network file. Otherwise, upload modules from model file and weights from weights
        # files given by the weights list file and build a neural network from scratch
        if model == "" and weights_list == "":

            nn_file = open(neural_net, 'r')
            print '\rReading neural network from file...'
            try:
                modules = json.load(nn_file)
            finally:
                nn_file.close()

        else:

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

            # now turn the list modules into a Python dictionary so we can access each module using the
            # module name as key (this makes index 0 dissapear and shifts all other list indexes by 1)
            # Therefore, the key to the dictionary 'modules' is now the name of the LSNM module
            # The indexes of each module list are as follows:
            # 0: module's X dimension (number of columns)
            # 1: module's Y dimension (number of rows)
            # 2: activation rule (neural population equation) or 'clamp' (constant value)
            # 3: Wilson-Cowan parameter 'threshold'
            # 4: Wilson-Cowan parameter 'Delta'
            # 5: Wilson-Cowan parameter 'delta'
            # 6: Wilson-Cowan parameter 'K'
            # 7: Wilson-Cowan parameter 'N'
            # 8: A python list of lists of X x Y elements containing the following elements:
            #     0: neural activity of current unit
            #     1: Sum of all inputs to current unit
            #     2: Sum of excitatory inputs to current unit
            #     3: Sum of inhibitory inputs to current unit
            #     4: a Python list of lists containing all outgoing connections arising from current unit, There are as
            #        many elements as outgoing connection weights and each element contains the following:
            #         0: destination module (where is the connection going to)
            #         1: X coordinate of location of destination unit
            #         2: Y coordinate of location of destination unit
            #         3: Connection weight
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

                            # now we decide whether the weights will be multiplied by a random amount
                            # varying between that amount and 1.0 in order to generate a new subject
                            if generateSubject == True:
                                connectionWeight = destination[1] * random.uniform(subject_variation, 1)
                            else:
                                connectionWeight = destination[1]

                            modules[origin_module][8][connection[0][0]-1][connection[0][1]-1][4].append (
                                [destination_module,        # insert name of destination module
                                 destination[0][0]-1,         # insert x coordinate of destination unit
                                 destination[0][1]-1,         # insert y coordinate of destination unit
                                 connectionWeight])           # insert connection weight 
                            synapse_count += 1

        # the following files store values over time of all units (electrical activity,
        # synaptic activity, to output data files in text format
        fs_neuronal   = []
        fs_abs_syn    = []
        fs_signed_syn = []
        
        # open one output file per module to record electrical, absolute synaptic activity (used for fMRI BOLD),
        # and signed synaptic activity (used for MEG source activity computation)
        for module in modules.keys():
            # open one output file per module
            fs_neuronal.append(open(module + '.out', 'w'))
            fs_abs_syn.append(open(module + '_abs_syn.out', 'w'))
            fs_signed_syn.append(open(module + '_signed_syn.out', 'w'))

        # create a dictionary so that each module name is associated with one output file
        fs_dict_neuronal  = dict(zip(modules.keys(), fs_neuronal))
        fs_dict_abs_syn   = dict(zip(modules.keys(), fs_abs_syn))
        fs_dict_signed_syn= dict(zip(modules.keys(), fs_signed_syn))

        # open the file with the experimental script and store the script in a string
        with open(script) as s:
            experiment_script = s.read()

        # open a file where we will dump the whole data structure (model and weights) in case it needs
        # to be used later, for inpection and/or visualization of neural network. We chose to use JSON
        # for this, due to its interoperability with other computer languages and other operating
        # systems. 
        nn_file = open(neural_network, 'w')
        print '\rSaving neural network to file...'
        try:
            json.dump(modules, nn_file)
        finally:
            nn_file.close()
            
        # initialize timestep counter for LSNM timesteps
        t = 0

        # import the experimental script given by user's script file
        exec(experiment_script)
        
        # define length of TVB simulation in ms
        TVB_simulation_length = LSNM_simulation_time * timesteps_to_time_constant

        # initialize number of timesteps for simulation
        sim_percentage = 100.0/LSNM_simulation_time

        # run the simulation for the number of timesteps given
        print '\rRunning simulation...'        
        
        if useTVBConnectome:
            # the following 'for loop' is the main loop of the TVB simulation with the parameters
            # defined above. Note that the LSNM simulator is literally embedded into the TVB
            # simulation and both run concurrently, timestep by timestep.
            for raw in sim(simulation_length=TVB_simulation_length):
            
                # convert current TVB connectome electrical activity to a numpy array 
                RawData = numpy.array(raw[0][1])
            
                # let the user know the percentage of simulation that has elapsed
                self.notifyProgress.emit(int(round(t*sim_percentage,0)))

                # check script to see if there are any event to be presented to the LSNM
                # network at current timestep t
                current_event=simulation_events.get(str(t))

                # then, introduce the event (if any was found)!
                # Note that 'modules' is defined within 'sim.py', whereas 'script_params' is
                # defined within the simulation script uploaded at runtime
                if current_event is not None:
                    current_event(modules, script_params)

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

                            
                # the following calculates (and integrates) synaptic activity at each TVB node
                # at the current timestep
                for tvb_node in range(TVB_number_of_nodes):

                    # rectifies or 'clamps' current tvb values to edges [0,1]
                    current_tvb_neu=np.clip(raw[0][1], 0, 1)
                
                    # extract TVB node numbers that are conected to TVB node above
                    tvb_conn = np.nonzero(white_matter.weights[tvb_node])
                    # extract the numpy array from it
                    tvb_conn = tvb_conn[0]

                    # build a numpy array of weights from TVB connections to the current TVB node
                    wm = white_matter.weights[tvb_node][tvb_conn]

                    # build a numpy array of origin TVB nodes connected to current TVB node
                    tvb_origin_node = raw[0][1][0][tvb_conn]

                    # clips node value to edges of interval [0, 1]
                    tvb_origin_node = np.clip(tvb_origin_node, 0, 1)
                
                    # do the following for each white matter connection to current TVB node:
                    # multiply all incoming connection weights times the value of the corresponding
                    # node that is sending that connection to the current TVB node
                    for cxn in range(tvb_conn.size):

                        # update synaptic activity in excitatory population, by multiplying each
                        # incoming connection weight times the value of the node sending such
                        # connection
                        current_tvb_abs_syn[0][tvb_node]    += wm[cxn] * tvb_origin_node[cxn][0]
                        current_tvb_signed_syn[0][tvb_node] += wm[cxn] * tvb_origin_node[cxn][0]

                    # now, add the influence of the local (within the same node) connectivity
                    # onto the synaptic activity of the current node, excitatory population, USING ABSOLUTE
                    # VALUES OF SYNAPTIC ACTIVITIES (for BOLD computation).
                    current_tvb_abs_syn[0][tvb_node] += w_ee * current_tvb_neu[0][tvb_node] + w_ie * current_tvb_neu[1][tvb_node]

                    # ... but also do a sum of synaptic activities using the sign of the synaptic activity (for MEG
                    # computation:
                    current_tvb_signed_syn[0][tvb_node] += w_ee * current_tvb_neu[0][tvb_node] - w_ie * current_tvb_neu[1][tvb_node]

                    # now, update synaptic activity in inhibitory population
                    # Please note that we are assuming that there are no incoming connections
                    # to inhibitory nodes from other nodes (in the Virtual Brain nodes).
                    # Therefore, only the local (within the same node) connections are
                    # considered (store both aboslute synaptic and signed synaptic):
                    current_tvb_abs_syn[1][tvb_node]    += w_ii * current_tvb_neu[1][tvb_node] + w_ei * current_tvb_neu[0][tvb_node]
                    current_tvb_signed_syn[1][tvb_node] += w_ei * current_tvb_neu[0][tvb_node] - w_ii * current_tvb_neu[1][tvb_node] 
    
                
                # the following 'for loop' goes through each LSNM module that is 'embedded' into The Virtual
                # Brain, and adds the product of each TVB -> LSNM unit value times their respective
                # connection weight (provided by white matter tract weights) to the sum of excitatory
                # activities of each embedded LSNM unit. THIS IS THE STEP
                # WHERE THE INTERACTION BETWEEN LSNM AND TVB HAPPENS. THAT INTERACTION GOES IN BOTH DIRECTIONS,
                # I.E., TVB -> LSNM and LSNM -> TVB. There is a constant called "FEEDBACK" that has to be
                # set to TRUE at the beginning of this file for the connections TVB->LSNM to occur.
                # Please note that whereas the previous 'for loop' goes through the network updating
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

                                    # extract value of TVB node
                                    value = RawData[0, tvb_conn[i]]
                                    value = value[0]
                                    # clips TVB node value to edges of interval [0, 1]
                                    value = max(value, 0)
                                    value = min(value, 1)
                                
                                    # calculate an incoming weight by applying a gain into the LSNM unit.
                                    # the gain applied is a random number with a gaussian distribution
                                    # centered around the value of lsnm_tvb_gain
                                    weight = wm[i] * rdm.gauss(lsnm_tvb_gain,lsnm_tvb_gain/4)
                                    value_x_weight = value * weight
                        
                                    # ... and add the incoming value_x_weight to the summed synaptic
                                    # activity of the current unit
                                    if value_x_weight > 0:
                                        modules[m][8][x][y][2] += value_x_weight
                                    else:
                                        modules[m][8][x][y][3] += value_x_weight
                                    # ... but store the total of inputs separately as well
                                    modules[m][8][x][y][1] += value_x_weight

                                    # And modify the connectome node that is providing the current
                                    # connecting weight i (but only if the 'feedback' flag is TRUE)
                                    if FEEDBACK:
                                        raw[0][1][0][tvb_conn[i]][0] += modules[m][8][x][y][0] * wm[i] * TVB_global_coupling_strength
                                
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
                                # now calculate and write out absolute sum of synaptic activity (for fMRI)...
                                abs_syn = modules[m][8][x][y][2] + abs(modules[m][8][x][y][3])
                                fs_dict_abs_syn[m].write(repr(abs_syn) + ' ')
                                # ... and calculate and write out signed sum of synaptic activity (for MEG):
                                signed_syn = modules[m][8][x][y][2] + modules[m][8][x][y][3]
                                fs_dict_signed_syn[m].write(repr(signed_syn) + ' ')
                                # ...finally, reset synaptic activity (but not neural activity).
                                modules[m][8][x][y][2] = 0.0
                                modules[m][8][x][y][3] = 0.0                            

                        
                    # finally, insert a newline character so we can start next set of units on a
                    # new line
                    fs_dict_neuronal[m].write('\n')
                    fs_dict_abs_syn[m].write('\n')
                    fs_dict_signed_syn[m].write('\n')

                # also write neural and synaptic activity of all TVB nodes to output files at
                # the current
                # time step, but ONLY IF a given number of timesteps has elapsed (integration
                # interval)
                if ((LSNM_simulation_time + t) % synaptic_interval) == 0:
                    # append the current TVB node electrical activity to array
                    tvb_elec.append(current_tvb_neu)
                    # append current synaptic activity array to synaptic activity timeseries
                    tvb_abs_syna.append(current_tvb_abs_syn)
                    tvb_signed_syna.append(current_tvb_signed_syn)
                    # reset TVB synaptic activity, but not TVB neuroelectrical activity
                    current_tvb_abs_syn    = [ [0.0]*TVB_number_of_nodes for _ in range(2) ]
                    current_tvb_signed_syn = [ [0.0]*TVB_number_of_nodes for _ in range(2) ]

            
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

        elif useTVBConnectome == False :   # ... if not using TVB connectome...

            for t in range(LSNM_simulation_time):

                # let the user know the percentage of simulation that has elapsed
                self.notifyProgress.emit(int(round(t*sim_percentage,0)))

                # check script to see if there are any event to be presented to the LSNM
                # network at current timestep t
                current_event=simulation_events.get(str(t))

                # then, introduce the event (if any was found)!
                # Note that 'modules' is defined within 'sim.py', whereas 'script_params' is
                # defined within the simulation script uploaded at runtime
                if current_event is not None:
                    current_event(modules, script_params)

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
                                # now calculate and write out absolute sum of synaptic activity (for fMRI)...
                                abs_syn = modules[m][8][x][y][2] + abs(modules[m][8][x][y][3])
                                fs_dict_abs_syn[m].write(repr(abs_syn) + ' ')
                                # ... and calculate and write out signed sum of synaptic activity (for MEG):
                                signed_syn = modules[m][8][x][y][2] + modules[m][8][x][y][3]
                                fs_dict_signed_syn[m].write(repr(signed_syn) + ' ')
                                # ...finally, reset synaptic activity (but not neural activity).
                                modules[m][8][x][y][2] = 0.0
                                modules[m][8][x][y][3] = 0.0                            

                        
                    # finally, insert a newline character so we can start next set of units on a
                    # new line
                    fs_dict_neuronal[m].write('\n')
                    fs_dict_abs_syn[m].write('\n')
                    fs_dict_signed_syn[m].write('\n')

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
                                    
                
        # be safe and close output files properly
        for f in fs_neuronal:
            f.close()
        for f in fs_abs_syn:
            f.close()
        for f in fs_signed_syn:
            f.close()
        
        if useTVBConnectome == True:
            # convert electrical and synaptic activity of TVB nodes into numpy arrays
            TVB_elec        = numpy.array(tvb_elec)
            TVB_abs_syna    = numpy.array(tvb_abs_syna)
            TVB_signed_syna = numpy.array(tvb_signed_syna) 

            # now, save the TVB electrical and synaptic activities to separate files
            numpy.save("tvb_neuronal.npy", TVB_elec)
            numpy.save("tvb_abs_syn.npy", TVB_abs_syna)
            numpy.save("tvb_signed_syn.npy", TVB_signed_syna)
            
        end_time = time_module.asctime(time_module.localtime(time_module.time()))
        
        # Finally (finally), save simulation data to a log file
        # ...more data to be added later, as needed
        with open(log_file, 'w') as f:
            f.write('* Simulation Start Time: ' + start_time)
            f.write('\n* Simulation End Time: ' + end_time)
            f.write('\n* Simulation duration (timesteps): ' + str(LSNM_simulation_time))
            f.write('\n* Model description used: ' + model)
            f.write('\n* Weights list used: ' + weights_list)
            f.write('\n* Neural net used: ' + neural_net)
            f.write('\n* Simulation script used: ' + script)            
            f.write('\n* Were weights changed to generate a new subject? ')
            if generateSubject == True:
                f.write('YES, weights multiplied by [' + str(subject_variation) + ', 1]')
            else:
                f.write('NO')
            f.write('\n* Was TVB Connectome used in the simulation? ')
            if useTVBConnectome == True:
                f.write('YES')
            else:
                f.write('NO')

        print '\rSimulation Finished.'
        print '\rOutput data files saved.'
        print '\rEnd Time: ', end_time

        

        
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


