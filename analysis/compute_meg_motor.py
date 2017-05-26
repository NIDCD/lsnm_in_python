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
#   This file (megVisual.py) was created on May 15, 2017.
#
#   Based on Sanz-Leon et al (2015) and Sarvas (1987) and on TVB's monitors.py
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on May 15 2017  
# **************************************************************************/

# compute_meg_motor.py
#
# Calculate and plot MEG signal at source locations based on data from motor

import numpy as np
import matplotlib.pyplot as plt

# specify indexes of tvb nodes with direct connections to M1
#m1_connected = [655,658,659,660,661,662,664,665,666,667,668,669,718,720,721,722,723,725,727,
#                728,729,730,732,733,734,735,760,763,764,767,768,791,792,795]
m1_connected = [662]



print m1_connected[0:len(m1_connected)]

#mu_0 = 1.25663706e-6 # mH/mm

# hypothesized Talairach coordinates of LSNM brain regions
#m1_loc = [18, -88, 8]

# initialize source positions
#r_0 = [v1_loc, v4_loc, it_loc, pf_loc] 

#initialize vector from sources to sensor
#Q = simulator.connectivity.orientations

#centre = numpy.mean(r_0, axis=0)[numpy.newaxis, :]
#radius = 1.01 * max(numpy.sqrt(numpy.sum((r_0 - centre)**2, axis=1)))

# Load M1 synaptic activity data files into a numpy array
exm1 = np.loadtxt('exm1_signed_syn.out')

# Load TVB synaptic activity data files into a numpy array
tvb = np.load('tvb_signed_syn.npy')
print 'Shape of connectome meg array before reshaping: ', tvb.shape
tvb_sliced = tvb[:, 0, m1_connected[0:len(m1_connected)], 0]
print 'Shape of connectome MEG array after slicing: ', tvb_sliced.shape

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = exm1.shape[0]

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

# add all units within each region (only M1 at this point) together across space to calculate
# MEG source dynamics in each brain region
m1 = np.sum(exm1, axis = 1)

# increase font size
plt.rcParams.update({'font.size': 30})

# Set up figure to plot MEG source dynamics
plt.figure(1)

plt.suptitle('SIMULATED MEG SOURCE DYNAMICS IN M1')

ax1=plt.subplot()
ax1.set_ylim([22, 30])

# Plot MEG signal
m1_plot=plt.plot(t, m1)

# Set up second figure to plot MEG source dynamics in nodes directly connected to M1
plt.figure(2)

plt.suptitle('SIMULATED MEG SOURCE DYNAMICS IN nodes connected to M1')

ax2=plt.subplot()

# Plot MEG signal
m1_connected_plot=plt.plot(t, tvb_sliced)


# Show the plot on the screen
plt.show()
