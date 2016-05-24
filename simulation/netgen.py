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
#   This file (netgen.py) was created on May 20 2016.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 24 2016
#
#   Based on computer code originally developed by Malle Tagamets and
#   Barry Horwitz (Tagamets and Horwitz, 1998)
# **************************************************************************/

# netgen.py
#
# Reads description of connecton weights among two simulated brain regions
# and generates all the connecton weights between those regions.
# The input, a weight file with extension "w", is given as argument when script
# is executed.
# The ouput is a file of the same name as the input file but with extension
# "ws".
# Only one weight file is processed at a time. If need to process all weight
# files in a directory, you might want to use the following bash script on a
# Unix or Linux terminal:
# for file in *.ws ;
#     do python netgen.py $file ;
# done

import sys

# import regular expression modules (useful for reading weight files)
import re

# import random function modules
import random as rdm

import time as time_module

# First, open file given in the command line (passed as argument)
with open(sys.argv[1], 'r') as f:
    try:
        contents = f.read()
        weights_file = sys.argv[1][:-1]
        print 'Generating weight file', weights_file, '...'

    except IOError:
        print 'Ouch!: For some reason I was unable read your input file (*.ws)'

# intitialize string that will contain weight files:
weights_string = []

# Scan the input weight file to find parameters of connection weights among brain regions, and save those
# parameters to relevant variables
parameters = re.match(r'(.*) (.*) SV I\((.*) (.*)\) O\((.*) (.*)\) F\((.*) (.*)\) (.*) (.*) Offset', contents)
InSet = parameters.group(1)
OutSet = parameters.group(2)
ix      = parameters.group(3)
iy      = parameters.group(4)
ox      = parameters.group(5)
oy      = parameters.group(6)
fx      = parameters.group(7)
fy      = parameters.group(8)
seed    = parameters.group(9)
pctzero = parameters.group(10)

# Now scan input weights file to find the actual weights and standard error of those weights (base and scale),
# and save all those weights to relevant python lists.
weights = re.findall(r'[+-]?\d+\.\d+:', contents) 
errors  = re.findall(r':\d+.\d+', contents)

# initialize base and scale
base = []
scale = []

# now, get rid of ":" in base and scale and convert base and scale to float
for b in weights:
    b = float(b[:-1])
    base.append(b)

for s in errors:
    s = float(s[1:]) 
    scale.append(s*2)

# convert numeric strings to int prior to making operations with them
ix = int(ix)
iy = int(iy)
ox = int(ox)
oy = int(oy)
fx = int(fx)
fy = int(fy)
seed = int(seed)
pctzero = float(pctzero)

istartx = 0
istarty = 0

ostartx = 0
ostarty = 0

idx = 1
idy = 1
odx = 1
ody = 1

# grab date and time from the system:
start_time = time_module.asctime(time_module.localtime(time_module.time()))

# Update final weights string with date and network information
weights_string  = "% " + start_time + "\n\n"
weights_string += "% Input Layer: "  + "(" + str(ix) + ", " + str(iy) + ")" + "\n"
weights_string += "% Output Layer: " + "(" + str(ox) + ", " + str(oy) + ")" + "\n"
weights_string += "% Fanout Size: " + "(" + str(fx) + ", " + str(fy) + ")" + "\n\n"
weights_string += "Connect(" + InSet + ", " + OutSet + ")  {\n"

row=0
col=0

# Initialize random number generator seed using system time
rdm.seed()

for i in range(istartx, ix, idx):
    for j in range(istarty, iy, idy):
        weights_string += "  From:  (" + str(i+1) + ", " + str(j+1) + ")  {\n"

        row = (ostartx - fx / 2 + ox) % ox
        col = (ostarty - fy / 2 + oy) % oy

        k = 0
        n_weights = 0

        for orow in range(0, fx, 1):
            for ocol in range(0, fy, 1):
                
                outx = (row + orow) % ox
                outy = (col + ocol) % oy

                x = rdm.random()
        
                if (x < pctzero) or (base[k] == 0.0):
                    weights_string += "    |              | "
                else:
                    x = rdm.random() - 0.5
                    x = base[k] + x * scale[k]
                    x = "{:.6f}".format(x)
                    weights_string += "    ([ " + str(outx+1) + ", " + str(outy+1) + "]  " + str(x) + ") "
                    n_weights = n_weights + 1

                k = k + 1
            weights_string += "\n"

        # make sure we have at least one outgoing set of weights
        if (pctzero < 1.0 and n_weights == 0):
            k = 0
            outx = i / ox
            outy = j / oy
            x = rdm.random() - 0.5
            x = base[k] + x * scale[k]
            x = "{:.6f}".format(x)
            weights_string += "    ([ " + str(outx+1) + ", " + str(outy+1) + "]  " + str(x) + ") "
            
        weights_string += "  }\n"

        ostarty = ostarty + ody

    ostartx = ostartx + odx

# insert a final closing curly bracket after the final set of weights
weights_string += "}\n"

# now, create a new weight file to write the generated weights to...
with open(weights_file, 'w') as f:
    try:
      f.write(weights_string)
    except IOError:
        print 'Ouch!: For some reason I cannot write to the output file (*.w)'


