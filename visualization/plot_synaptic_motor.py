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
#   This file (plot_synaptic_motor.py) was created on May 15, 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 15 2017  
# **************************************************************************/

# plot_synaptic_motor.py
#
# Plot synaptic activity data from motor model

import numpy as np
import matplotlib.pyplot as plt

# Load M1 synaptic activity data files into a numpy array
exm1 = np.loadtxt('exm1_abs_syn.out')
inm1 = np.loadtxt('inm1_abs_syn.out')

# Extract number of timesteps from one of the matrices
timesteps = exm1.shape[0]
print timesteps

# the following variable defines the timesteps we will see in the resulting plot
# we also convert the number of timesteps to seconds by multiplying by 50 and dividng by 1000
ts_to_plot = 600
x_lim = ts_to_plot * 50. / 1000.

# Construct a numpy array of timesteps (data points provided in data file)
# to convert from timesteps to time in seconds we do the following:
# Each simulation time-step equals 5 milliseconds
# However, we are recording only once every 10 time-steps
# Therefore, each data point in the output files represents 50 milliseconds.
# Thus, we need to multiply the datapoint times 50 ms...
# ... and divide by 1000 to convert to seconds
t = np.linspace(0, (ts_to_plot-1) * 50.0 / 1000., num=ts_to_plot)

# add all units within each region (V1, IT, and D1) together across space to calculate
# synaptic activity in each brain region
m1 = np.sum(exm1 + inm1, axis = 1)

# Set up plot
plt.figure(1)

#plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# increase font size
plt.rcParams.update({'font.size': 30})

ax1=plt.subplot()

ax1.set_ylim([75, 92])

# Plot V1 module
plt.plot(t, m1[0:ts_to_plot], color='yellow', linewidth=2)

plt.gca().set_axis_bgcolor('black')

plt.xlabel('Time (s)')

plt.tight_layout()

# Show the plot on the screen
plt.show()

