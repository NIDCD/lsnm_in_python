#!/usr/bin/python
#
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
#   This file (Auditory_UI.py) was created on July 21, 2015.
#
#
#   Author: John Gilbert. Last updated by John Gilbert on July 21 2015
#
# **************************************************************************/

# Auditory_UI.py
#
# GUI to make and test new auditory inputs

import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np

 
class Main(QtGui.QDialog):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.initUI()
    
    def initUI(self): 
	# Create Time Dialog Boxes
	self.tdiff = QtGui.QLabel('Time',self)
        self.tdiff_int = QtGui.QLabel('Int', self)
	self.tdiff_low = QtGui.QLabel('Low',self)
	self.tdiff_high = QtGui.QLabel('High',self)
        self.tdiff_low_in = QtGui.QLineEdit('200',self)
	self.tdiff_int_in = QtGui.QLineEdit('10',self)
	self.tdiff_high_in = QtGui.QLineEdit('300',self)

	
	# Move/Resize Time Boxes	
	self.tdiff.move(100,10)        
	self.tdiff_int.move(110, 30)
	self.tdiff_low.move(30, 30)
	self.tdiff_high.move(190,30)
	self.tdiff_low_in.move(20, 45)
	self.tdiff_low_in.resize(60,25)
        self.tdiff_int_in.move(100, 45)
	self.tdiff_int_in.resize(60,25)
	self.tdiff_high_in.move(180, 45)
	self.tdiff_high_in.resize(60,25)

	# MATCH CASES

	# Create Frequency Dialog Boxes and move/resize
	self.freq = QtGui.QLabel('Frequency',self)
        self.lowf = QtGui.QLabel('Low', self)
	self.intf = QtGui.QLabel('Int', self)
        self.highf = QtGui.QLabel('High', self)
        self.lowf_in = QtGui.QLineEdit('41',self)
	self.intf_in = QtGui.QLineEdit('1',self)        
	self.highf_in = QtGui.QLineEdit('46',self)

	
	self.freq.move(100,85)
	self.lowf.move(30, 105)
        self.intf.move(110, 105)
        self.highf.move(190, 105)
        self.lowf_in.move(20, 120)
        self.intf_in.move(100, 120)
        self.highf_in.move(180, 120)
	self.lowf_in.resize(60,25)
	self.intf_in.resize(60,25)
	self.highf_in.resize(60,25)

	# MISMATCH CASES

	# Create Frequency Dialog Boxes and move/resize
	self.mfreq = QtGui.QLabel('Mismatch Frequency',self)
        self.mlowf = QtGui.QLabel('Low', self)
	self.mintf = QtGui.QLabel('Int', self)
        self.mhighf = QtGui.QLabel('High', self)
        self.mlowf_in = QtGui.QLineEdit('63',self)
	self.mintf_in = QtGui.QLineEdit('1',self)        
	self.mhighf_in = QtGui.QLineEdit('68',self)

	
	self.mfreq.move(300,85)
	self.mlowf.move(280, 105)
        self.mintf.move(360, 105)
        self.mhighf.move(440, 105)
        self.mlowf_in.move(270, 120)
        self.mintf_in.move(350, 120)
        self.mhighf_in.move(430, 120)
	self.mlowf_in.resize(60,25)
	self.mintf_in.resize(60,25)
	self.mhighf_in.resize(60,25)

	# Create LineEdit for Peak Time Duration	
	#self.tpeak = QtGui.QLabel('Peak Time Steps', self)
        #self.tpeak.move(20, 140)
	#self.tpeak_in = QtGui.QLineEdit('10',self)
        #self.tpeak_in.move(150, 142)

	# Create Checkbox for Match/Mismatch Switch
	cb = QtGui.QCheckBox('Match', self)
	self.match = 1
        cb.move(20, 160)
        cb.toggle()
        cb.stateChanged.connect(self.changeMatch)

	# Create Checkbox for Tone/Tonal Contour Switch
	cb = QtGui.QCheckBox('Tonal Contour', self)
	self.TC = 1
        cb.move(20, 190)
        cb.toggle()
        cb.stateChanged.connect(self.changeTC)

	# Create LineEdit for Repetitions
	self.repeat = QtGui.QLabel('Repetitions', self)
        self.repeat_in = QtGui.QLineEdit('4',self)
	self.repeat.move(190,160)
	self.repeat_in.move(160, 155)
	self.repeat_in.resize(25,25)

	# Create LineEdit for Attention
	self.attn = QtGui.QLabel('Attention', self)
        self.attn_in = QtGui.QLineEdit('0.30',self)
	self.attn.move(190,190)
	self.attn_in.move(160, 185)
	self.attn_in.resize(25,25)
	
	# Create Connect Button
        self.pb = QtGui.QPushButton('Accept',self)
	self.pb.clicked.connect(self.button_click)
	self.pb.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.pb.move(20, 225)
	self.pb.resize(100,25)
	
	# Create Quit Button
	self.close = QtGui.QPushButton('Close',self)
	self.close.clicked.connect(QtCore.QCoreApplication.instance().quit)
	self.close.move(150,225)
	self.close.resize(100,25)

	# Create Save File Button
        
	# Set Window Geometry
        self.setFixedSize(270,270)
        self.setWindowTitle("Auditory Input Generator")
        self.setWindowIcon(QtGui.QIcon(""))
        
	# Text File Defaults
	self.Start_text ='    for x in range(modules[\'mgns\'][0]):\n        for y in range(modules[\'mgns\'][1]):\n	    modules[\'mgns\'][8][x][y][0] = 0.0\n	    modules[\'ea1u\'][8][x][y][0] = 0.0\n	    modules[\'ea1d\'][8][x][y][0] = 0.0\n	    modules[\'ia1u\'][8][x][y][0] = 0.0\n	    modules[\'ia1d\'][8][x][y][0] = 0.0\n\n'
         
	self.ISI_text = '\n    # INTERSTIMULUS INTERVAL\n    for x in range(modules[\'mgns\'][0]):\n        for y in range(modules[\'mgns\'][1]):\n	    modules[\'mgns\'][8][x][y][0] = 0.0\n\n'

	self.ITI_text_1 = '\n    # ################# INTERTRIAL INTERVAL STARTS #############\n    # reset activity in all units at the end of a trial,\n    # just to make sure any \'spontaneous activity\' is dealt with\n    for x in range(modules[\'mgns\'][0]):\n        for y in range(modules[\'mgns\'][1]):\n            modules[\'mgns\'][8][x][y][0] = 0.0\n	    modules[\'ea1u\'][8][x][y][0] = 0.0\n	    modules[\'ea1d\'][8][x][y][0] = 0.0\n	    modules[\'ia1u\'][8][x][y][0] = 0.0\n	    modules[\'ia1d\'][8][x][y][0] = 0.0\n	    modules[\'ea2u\'][8][x][y][0] = 0.0\n	    modules[\'ea2d\'][8][x][y][0] = 0.0\n	    modules[\'ea2c\'][8][x][y][0] = 0.0\n	    modules[\'ia2u\'][8][x][y][0] = 0.0\n	    modules[\'ia2d\'][8][x][y][0] = 0.0\n	    modules[\'ia2c\'][8][x][y][0] = 0.0\n'

	self.ITI_text_2 = '    for x in range(modules[\'estg\'][0]):\n        for y in range(modules[\'estg\'][1]):\n	    modules[\'estg\'][8][x][y][0] = 0.0\n	    modules[\'istg\'][8][x][y][0] = 0.0\n	    modules[\'exfs\'][8][x][y][0] = 0.0\n	    modules[\'infs\'][8][x][y][0] = 0.0\n	    modules[\'efd1\'][8][x][y][0] = 0.0\n	    modules[\'ifd1\'][8][x][y][0] = 0.0\n	    modules[\'efd2\'][8][x][y][0] = 0.0\n	    modules[\'ifd2\'][8][x][y][0] = 0.0\n	    modules[\'exfr\'][8][x][y][0] = 0.0\n	    modules[\'infr\'][8][x][y][0] = 0.0\n    # turn attention to \'LOW\', as the current trial has ended\n    modules[\'atts\'][8][0][0][0]  = 0.05\n    ################ INTERTRIAL INTERVAL ENDS #################\n\n'

	self.base_start = '    modules[\'mgns\'][8][0]['
	self.base_end = '][0] = '
	self.time_start = '\nif t == '


    # Accept Button       
    def button_click(self,lines):
        openFile = QtGui.QAction('Save', self)
	fname = QtGui.QFileDialog.getSaveFileName(self, 'Save Script file', '/home/intern2/documents/auditory_lsnm/')
        
        f = open(fname, 'w')    
                    
	def loops(self,ml,time,fr1,fr2,b,boo):	
	# Creates function to create the default loop from stim1 presentation to stim2 presentation
	# Inputs are
	# 1) Self
	# 2) ml, the number of loops that must be created
	# 3) time, the time vector to draw from
	# 4) fr1, the first frequency vector
	# 5) fr2, the second frequency vector to draw from
	# 6) b, a 1x4 vector to input the order for drawing from the lines vector based on whether the TC is upsweep or downsweep and a match or mismatch.
	# 7) boo, a true/false variable to indicate whether or not it is an up or downsweep.   

	    def lines(self,x,a):
	        # Create function to print the default lines
	        # Inputs are self, the 3rd value in modules to increment, and +/-1 for the direction
                f.write(self.base_start + str(x-a) + self.base_end + '0.0\n')
	        f.write(self.base_start + str(x) + self.base_end + '0.0\n')
                f.write(self.base_start + str(x) + self.base_end + '1.0\n')
                f.write(self.base_start + str(x+a) + self.base_end + '1.0\n')
	    
	    f.write(self.base_start + str(fr1[0]) + self.base_end + '1.0\n')
            f.write(self.base_start + str(fr1[1]) + self.base_end + '1.0\n')

	    for x in range(1,ml):
	        f.write(self.time_start + str(time[x]) + ':\n')
	        if boo:
		    lines(self,fr1[x],b[0])
		else:
		    lines(self,fr1[x],b[1])

	    for y in range(ml):
	    	f.write(self.time_start + str(time[x+y+1]) + ':\n')
	    	if boo:
		    lines(self,fr2[y],b[2])
		else:
		    lines(self,fr2[y],b[3])
	
	# TIME IN
	txt_lowt = int(self.tdiff_low_in.text())
	txt_intt = int(self.tdiff_int_in.text())
	txt_hight = int(self.tdiff_high_in.text())

	# FREQUENCY IN
	txt_lowf = int(self.lowf_in.text())
	txt_intf = int(self.intf_in.text())
	txt_highf = int(self.highf_in.text())

	# MISMATCH FREQ IN
	txt_mlowf = int(self.mlowf_in.text())
	txt_mintf = int(self.mintf_in.text())
	txt_mhighf = int(self.mhighf_in.text())

	# OTHER INPUTS
	#txt_tpeak = int(self.tpeak_in.text())
	# Repetitions
	txt_reps = int(self.repeat_in.text())
	# Attention
	txt_attn = float(self.attn_in.text()) 


	# CREATE TIME INTERVALS
	time_int = range(txt_lowt,txt_hight+txt_intt,txt_intt)
	
	# CREATE MATCH FREQUENCY INTERVALS
	if txt_highf > txt_lowf:	
	    freq_int = range(txt_lowf,txt_highf,txt_intf)
	    freq_int2 = range(txt_highf,txt_lowf,-txt_intf)
	else:
	    freq_int = range(txt_lowf,txt_highf,-txt_intf)
	    freq_int2 = range(txt_highf,txt_lowf,txt_intf)
	


	min_len = min([len(freq_int),len(time_int)])

	# CHECK IF MATCH TO DETERMINE LENGTHS USED
	if self.match == 0:
	    # CREATE MISMATCH FREQUENCY INTERVALS
	    if txt_mhighf > txt_mlowf:	
	        freq_mint = range(txt_mlowf,txt_mhighf,txt_mintf)
	        freq_mint2 = range(txt_mhighf,txt_mlowf,-txt_mintf)
	    else:
	        freq_mint = range(txt_mlowf,txt_mhighf,-txt_mintf)
	        freq_mint2 = range(txt_mhighf,txt_mlowf,txt_mintf)
	    min_len = min([len(freq_mint),len(time_int)])
	
	# CHECK UP OR DOWNSWEEP	
	if txt_highf > txt_lowf:
	    boo = True
	else:
	    boo = False

	# START OF TRIAL UPDATE (not repeated)
	f.write('if t == ' + str(txt_lowt-200) + ':\n')
	f.write(self.Start_text)

	# IF TONE:
	if self.TC == 0:
	    for k in range(txt_reps):
		
		# Start Text
		f.write(self.time_start + str(time_int[0]) + ':\n')
		f.write(self.base_start + str(txt_lowf) + self.base_end + '1.0\n')

		# INTERSTIMULUS INTERVAL TEXT
	    	f.write('if t == ' + str(time_int[0]+100) + ':\n')
	    	f.write(self.ISI_text)
		
		# Update Time
		time_int = [x+300 for x in time_int]
		f.write(' if t == ' + str(time_int[0]) + ':\n')

		# IF MATCH
		if self.match == 1:
		    f.write(self.base_start + str(txt_lowf) + self.base_end + '1.0\n')
		else:
		    f.write(self.base_start + str(txt_mlowf) + self.base_end + '1.0\n')

	        # INTERTRIAL INTERVAL TEXT
	        f.write('if t == ' + str(time_int[0] + 100) + ':\n')
	        f.write(self.ITI_text_1)
	        f.write(self.ITI_text_2)
	        f.write('    # turn attention to \'HI\', as the input stimulus is about to be presented\n    modules[\'atts\'][8][0][0][0] = ' + str(txt_attn) + '\n')
	        # Update Time
	        time_int = [x+400 for x in time_int]
		
	# IF TONAL CONTOUR
	else:
	# REPETITIONS
	    for k in range(txt_reps): 	
	
	        f.write(self.time_start + str(time_int[0]) + ':\n')

		loops(self,min_len,time_int,freq_int,freq_int2,[1,-1,-1,1],boo)
		
		# INTERSTIMULUS INTERVAL TEXT
	    	f.write('if t == ' + str(time_int[0]+100) + ':\n')
	    	f.write(self.ISI_text)
		
		# Update Time
		time_int = [x+300 for x in time_int]
		f.write(' if t == ' + str(time_int[0]) + ':\n')

		# IF MATCH
		if self.match == 1:
		    loops(self,min_len,time_int,freq_int,freq_int2,[1,-1,-1,1],boo)
		# ELSEIF MISMATCH		
		else:
		    loops(self,min_len,time_int,freq_mint,freq_mint2,[1,1,-1,-1],boo)
	
	        # INTERTRIAL INTERVAL TEXT
	        f.write('if t == ' + str(time_int[0] + 100) + ':\n')
	        f.write(self.ITI_text_1)
	        f.write(self.ITI_text_2)
	        f.write('    # turn attention to \'HI\', as the input stimulus is about to be presented\n    modules[\'atts\'][8][0][0][0] = ' + str(txt_attn) + '\n')
	        # Update Time
	        time_int = [x+400 for x in time_int]
        f.close()
    # Match Checkbox
    def changeMatch(self,state):
	# Functions to define states
	def mismatch_normal(self):
	    self.close.move(150,225)
            self.pb.move(20, 225)
	    self.mfreq.move(300,85)
	    self.mlowf.move(280, 105)
            self.mintf.move(360, 105)
            self.mhighf.move(440, 105)
            self.mlowf_in.move(270, 120)
            self.mintf_in.move(350, 120)
            self.mhighf_in.move(430, 120)
	def match_normal(self):
	    self.lowf.move(30, 105)
            self.intf.move(110, 105)
            self.highf.move(190, 105)
            self.lowf_in.move(20, 120)
            self.intf_in.move(100, 120)
            self.highf_in.move(180, 120)
	def match_tone(self):
            self.lowf_in.move(100, 105)
            self.intf_in.move(350, 520)
            self.highf_in.move(430, 520)

	    self.lowf.move(280, 505)
            self.intf.move(360, 505)
            self.highf.move(440, 505)
	def mismatch_tone(self):
            self.pb.move(20, 295)
	    self.close.move(150,295)
            self.mlowf_in.move(100, 240)
            self.mintf_in.move(350, 520)
            self.mhighf_in.move(430, 520)
	    self.mfreq.move(75,220)
	    self.mlowf.move(280, 505)
            self.mintf.move(360, 505)
            self.mhighf.move(440, 505)

	def mismatch_tc(self):
            self.pb.move(20, 295)
	    self.close.move(150,295)
            self.mlowf_in.move(20, 255)
            self.mintf_in.move(100, 255)
            self.mhighf_in.move(180, 255)
	    self.mfreq.move(75,220)
	    self.mlowf.move(30, 240)
            self.mintf.move(110, 240)
            self.mhighf.move(190, 240)

		

        if state == QtCore.Qt.Checked:
            self.match = 1
	    self.setFixedSize(270,270)
	    
	    # Match with Tonal Contour (Normal)
	    if self.TC == 1:
		match_normal(self)
		mismatch_normal(self)

	    # Match with Tone
	    else:
		match_tone(self)
		mismatch_normal(self)
		
        else:
            self.match = 0
	    self.setFixedSize(270,330)
	    # Tonal Contour with mismatch
	    if self.TC == 1:
		mismatch_tc(self)
		match_normal(self)
	    # Tone with Mismatch
	    else:
		match_tone(self)
		mismatch_tone(self)


    # Tonal Contour Checkbox
    def changeTC(self,state):
	# Functions to define states
	def mismatch_normal(self):
	    self.close.move(150,225)
            self.pb.move(20, 225)
	    self.mfreq.move(300,85)
	    self.mlowf.move(280, 105)
            self.mintf.move(360, 105)
            self.mhighf.move(440, 105)
            self.mlowf_in.move(270, 120)
            self.mintf_in.move(350, 120)
            self.mhighf_in.move(430, 120)
	def match_normal(self):
	    self.lowf.move(30, 105)
            self.intf.move(110, 105)
            self.highf.move(190, 105)
            self.lowf_in.move(20, 120)
            self.intf_in.move(100, 120)
            self.highf_in.move(180, 120)
	def match_tone(self):
            self.lowf_in.move(100, 105)
            self.intf_in.move(350, 520)
            self.highf_in.move(430, 520)

	    self.lowf.move(280, 505)
            self.intf.move(360, 505)
            self.highf.move(440, 505)
	def mismatch_tone(self):
            self.pb.move(20, 295)
	    self.close.move(150,295)
            self.mlowf_in.move(100, 240)
            self.mintf_in.move(350, 520)
            self.mhighf_in.move(430, 520)
	    self.mfreq.move(75,220)
	    self.mlowf.move(280, 505)
            self.mintf.move(360, 505)
            self.mhighf.move(440, 505)

	def mismatch_tc(self):
            self.pb.move(20, 295)
	    self.close.move(150,295)
            self.mlowf_in.move(20, 255)
            self.mintf_in.move(100, 255)
            self.mhighf_in.move(180, 255)
	    self.mfreq.move(75,220)
	    self.mlowf.move(30, 240)
            self.mintf.move(110, 240)
            self.mhighf.move(190, 240)

	# Check states
	if self.match == 1:
	    self.setFixedSize(270,270)
	    # Match with Tonal Contour
	    if state == QtCore.Qt.Checked:
	        self.TC = 1
		match_normal(self)
		mismatch_normal(self)
	    # Match with Tone
	    else:
	        self.TC = 0
		match_tone(self)
		mismatch_normal(self)
	else:
	    self.setFixedSize(270,330)
	    if state == QtCore.Qt.Checked:
		self.TC = 1
		match_normal(self)
		mismatch_tc(self)

	    else:
		self.TC = 0
		match_tone(self)
		mismatch_tone(self)
		
		

    def closeEvent(self, event):
	reply = QtGui.QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QtGui.QMessageBox.Yes | 
            QtGui.QMessageBox.No, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

def main():
    app = QtGui.QApplication(sys.argv)
    main= Main()
    main.show()
 
    sys.exit(app.exec_())
 
if __name__ == "__main__":
    main()
	

