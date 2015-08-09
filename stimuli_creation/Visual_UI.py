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
#   This file (Visual_UI.py) was created on July 21, 2015.
#
#
#   Author: John Gilbert. Last updated by John Gilbert on July 21 2015
#
# **************************************************************************/

# Visual_UI.py
#
# GUI to make and test new visual inputs

import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import numpy as np

xLoc = []
yLoc = []
nums = []
oper_letter = ''
varlist = []
store = []

newNum = 0.0
sumAll = 0.0
operator = ""
 
opVar = False
sumIt = 0
 
class Main(QtGui.QMainWindow):
 
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.initUI()
 
    def initUI(self):
	global nums
 
        b11 = QtGui.QPushButton("0,0",self)
        b11.move(10,30)

        b12 = QtGui.QPushButton("0,1",self)
        b12.move(40,30)

        b13 = QtGui.QPushButton("0,2",self)
        b13.move(70,30)

        b14 = QtGui.QPushButton("0,3",self)
        b14.move(100,30)

        b15 = QtGui.QPushButton("0,4",self)
        b15.move(130,30)

        b16 = QtGui.QPushButton("0,5",self)
        b16.move(160,30)

        b17 = QtGui.QPushButton("0,6",self)
        b17.move(190,30)

        b18 = QtGui.QPushButton("0,7",self)
        b18.move(220,30)

        b19 = QtGui.QPushButton("0,8",self)
        b19.move(250,30)

        b21 = QtGui.QPushButton("1,0",self)
        b21.move(10,60)

        b22 = QtGui.QPushButton("1,1",self)
        b22.move(40,60)

        b23 = QtGui.QPushButton("1,2",self)
        b23.move(70,60)

        b24 = QtGui.QPushButton("1,3",self)
        b24.move(100,60)

        b25 = QtGui.QPushButton("1,4",self)
        b25.move(130,60)

        b26 = QtGui.QPushButton("1,5",self)
        b26.move(160,60)

        b27 = QtGui.QPushButton("1,6",self)
        b27.move(190,60)

        b28 = QtGui.QPushButton("1,7",self)
        b28.move(220,60)

        b29 = QtGui.QPushButton("1,8",self)
        b29.move(250,60)

        b31 = QtGui.QPushButton("2,0",self)
        b31.move(10,90)

        b32 = QtGui.QPushButton("2,1",self)
        b32.move(40,90)

        b33 = QtGui.QPushButton("2,2",self)
        b33.move(70,90)

        b34 = QtGui.QPushButton("2,3",self)
        b34.move(100,90)

        b35 = QtGui.QPushButton("2,4",self)
        b35.move(130,90)

        b36 = QtGui.QPushButton("2,5",self)
        b36.move(160,90)

        b37 = QtGui.QPushButton("2,6",self)
        b37.move(190,90)

        b38 = QtGui.QPushButton("2,7",self)
        b38.move(220,90)

        b39 = QtGui.QPushButton("2,8",self)
        b39.move(250,90)

        b41 = QtGui.QPushButton("3,0",self)
        b41.move(10,120)

        b42 = QtGui.QPushButton("3,1",self)
        b42.move(40,120)

        b43 = QtGui.QPushButton("3,2",self)
        b43.move(70,120)

        b44 = QtGui.QPushButton("3,3",self)
        b44.move(100,120)

        b45 = QtGui.QPushButton("3,4",self)
        b45.move(130,120)

        b46 = QtGui.QPushButton("3,5",self)
        b46.move(160,120)

        b47 = QtGui.QPushButton("3,6",self)
        b47.move(190,120)

        b48 = QtGui.QPushButton("3,7",self)
        b48.move(220,120)

        b49 = QtGui.QPushButton("3,8",self)
        b49.move(250,120)

	b51 = QtGui.QPushButton("4,0",self)
        b51.move(10,150)

        b52 = QtGui.QPushButton("4,1",self)
        b52.move(40,150)

        b53 = QtGui.QPushButton("4,2",self)
        b53.move(70,150)

        b54 = QtGui.QPushButton("4,3",self)
        b54.move(100,150)

        b55 = QtGui.QPushButton("4,4",self)
        b55.move(130,150)

        b56 = QtGui.QPushButton("4,5",self)
        b56.move(160,150)

        b57 = QtGui.QPushButton("4,6",self)
        b57.move(190,150)

        b58 = QtGui.QPushButton("4,7",self)
        b58.move(220,150)

        b59 = QtGui.QPushButton("4,8",self)
        b59.move(250,150)
 
        b61 = QtGui.QPushButton("5,0",self)
        b61.move(10,180)

        b62 = QtGui.QPushButton("5,1",self)
        b62.move(40,180)

        b63 = QtGui.QPushButton("5,2",self)
        b63.move(70,180)

        b64 = QtGui.QPushButton("5,3",self)
        b64.move(100,180)

        b65 = QtGui.QPushButton("5,4",self)
        b65.move(130,180)

        b66 = QtGui.QPushButton("5,5",self)
        b66.move(160,180)

        b67 = QtGui.QPushButton("5,6",self)
        b67.move(190,180)

        b68 = QtGui.QPushButton("5,7",self)
        b68.move(220,180)

        b69 = QtGui.QPushButton("5,8",self)
        b69.move(250,180)

        b71 = QtGui.QPushButton("6,0",self)
        b71.move(10,210)

        b72 = QtGui.QPushButton("6,1",self)
        b72.move(40,210)

        b73 = QtGui.QPushButton("6,2",self)
        b73.move(70,210)

        b74 = QtGui.QPushButton("6,3",self)
        b74.move(100,210)

        b75 = QtGui.QPushButton("6,4",self)
        b75.move(130,210)

        b76 = QtGui.QPushButton("6,5",self)
        b76.move(160,210)

        b77 = QtGui.QPushButton("6,6",self)
        b77.move(190,210)

        b78 = QtGui.QPushButton("6,7",self)
        b78.move(220,210)

        b79 = QtGui.QPushButton("6,8",self)
        b79.move(250,210)

        b81 = QtGui.QPushButton("7,0",self)
        b81.move(10,240)

        b82 = QtGui.QPushButton("7,1",self)
        b82.move(40,240)

        b83 = QtGui.QPushButton("7,2",self)
        b83.move(70,240)

        b84 = QtGui.QPushButton("7,3",self)
        b84.move(100,240)

        b85 = QtGui.QPushButton("7,4",self)
        b85.move(130,240)

        b86 = QtGui.QPushButton("7,5",self)
        b86.move(160,240)

        b87 = QtGui.QPushButton("7,6",self)
        b87.move(190,240)

        b88 = QtGui.QPushButton("7,7",self)
        b88.move(220,240)

        b89 = QtGui.QPushButton("7,8",self)
        b89.move(250,240)

        b91 = QtGui.QPushButton("8,0",self)
        b91.move(10,270)

        b92 = QtGui.QPushButton("8,1",self)
        b92.move(40,270)

        b93 = QtGui.QPushButton("8,2",self)
        b93.move(70,270)

        b94 = QtGui.QPushButton("8,3",self)
        b94.move(100,270)

        b95 = QtGui.QPushButton("8,4",self)
        b95.move(130,270)

        b96 = QtGui.QPushButton("8,5",self)
        b96.move(160,270)

        b97 = QtGui.QPushButton("8,6",self)
        b97.move(190,270)

        b98 = QtGui.QPushButton("8,7",self)
        b98.move(220,270)

        b99 = QtGui.QPushButton("8,8",self)
        b99.move(250,270)

 
        DefaultE = QtGui.QPushButton("E",self)
	DefaultE.move(10,300)

        DefaultT = QtGui.QPushButton("T",self)
	DefaultT.move(60,300)

        DefaultO = QtGui.QPushButton("O",self)
	DefaultO.move(110,300)

        Defaultn = QtGui.QPushButton("n",self)
	Defaultn.move(180,300)

        DefaultU = QtGui.QPushButton("u",self)
	DefaultU.move(230,300)

	Reset = QtGui.QPushButton("RESET",self)
	Reset.move(50,360)
	Reset.resize(200,40)
        Reset.clicked.connect(self.reset)
	
	Accept = QtGui.QPushButton("Accept",self)
	Accept.move(50,420)
	Accept.resize(200,40)
	Accept.clicked.connect(self.accept)


 
        nums = [b11,b12,b13,b14,b15,b16,b17,b18,b19,b21,b22,b23,b24,b25,b26,b27,b28,b29,
		b31,b32,b33,b34,b35,b36,b37,b38,b39,b41,b42,b43,b44,b45,b46,b47,b48,b49,
		b51,b52,b53,b54,b55,b56,b57,b58,b59,b61,b62,b63,b64,b65,b66,b67,b68,b69,
		b71,b72,b73,b74,b75,b76,b77,b78,b79,b81,b82,b83,b84,b85,b86,b87,b88,b89,
		b91,b92,b93,b94,b95,b96,b97,b98,b99]
 
        defaults = [DefaultT, DefaultO, DefaultE, DefaultU, Defaultn]
 
        rest = []
 
        for i in nums:
            i.setStyleSheet("color:blue;")
	    i.setStyleSheet("background-color:none")
            i.clicked.connect(self.Nums)
	    i.resize(25,25)
 
        for i in defaults:
            i.setStyleSheet("color:red;")
	    i.resize(40,40)
	    i.clicked.connect(self.defaults)
 
         
             
#---------Window settings --------------------------------
         
        self.setGeometry(500,500,500,500)
        self.setFixedSize(300,500)
        self.setWindowTitle("Visual Input Generator")
        self.setWindowIcon(QtGui.QIcon(""))
        self.show()
    def defaults(self):
	global varlist	
	sender = self.sender()
	oper_letter = sender.text()
		
	if oper_letter == 'E':
	    varlist =  ['b54','b55','b56','b57','b58',
			'b74','b75','b76','b77','b78',
			'b94','b95','b96','b97','b98',
			'b64','b84']
	elif oper_letter == 'T':
	    varlist =  ['b41','b42','b43','b44','b45',
		        'b46','b47','b48','b49','b37',
		        'b38','b27','b28','b17','b18']
	elif oper_letter == 'O':
	    varlist = ['b54','b55','b56','b57','b58','b59',
			'b94','b95','b96','b97','b98','b99',
			'b64','b74','b84','b69','b79','b89']	
	elif oper_letter == 'n':
	    varlist = ['b34','b35','b36','b37','b38','b39',
		       'b44','b49','b54','b59','b64','b69','b74','b79']	
	elif oper_letter == 'u':
	    varlist = ['b34','b75','b76','b77','b78','b39',
		       'b44','b49','b54','b59','b64','b69','b74','b79']	
        
    
    def Nums(self):
        global xLoc
	global yLoc
        sender = self.sender()
	operator = sender.text()
	xLoc_sender = int(operator[0])
	yLoc_sender = int(operator[2])
	
	# If element already there (It is being unclicked
	if operator in store:
	    # Remove elements already there
	    store.remove(operator)
	    xLoc.remove(xLoc_sender)
	    yLoc.remove(yLoc_sender)
	    sender.setStyleSheet('background-color:none')

	else:    
	# Accept elements that are already there
	    store.append(operator)
	    xLoc.append(xLoc_sender)
	    yLoc.append(yLoc_sender)
            sender.setStyleSheet("background-color:red")

 
    def reset(self):
        global xLoc
	global yLoc
	global nums
	xLoc = []
	yLoc = []
        for i in nums:
	    i.setStyleSheet("background-color:none")

    def accept(self):
        global xLoc
	global yLoc
	global nums

	if len(varlist)>0:
	    for i in varlist:
	        print '    modules[\'lgns\'][8]['+str(int(i[1]))+']['+str(int(i[2]))+'][0] = 0.7'
		print '\n'
	else:
	    for j in range(len(xLoc)):
	        print '    modules[\'lgns\'][8]['+str(xLoc[j])+']['+str(yLoc[j])+'][0] = 0.7'
	print '\n' 	
    def update(self,oper_letter):


	if operator == '':
	    # Do Nothing
	    a = 0
	

	

def main():
    app = QtGui.QApplication(sys.argv)
    main= Main()
    main.show()
 
    sys.exit(app.exec_())
 
if __name__ == "__main__":
    main()

