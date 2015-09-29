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
#   This file (compute_BOLD_poisson_model.py) was created on April 17, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 11 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_BOLD_poisson_model.py
#
# Calculate and plot fMRI BOLD signal using a convolution with hemodynamic response
# function modeled as a poisson time-series, based on data from visual
# delay-match-to-sample simulation. It also saves the BOLD timeseries for each
# and all modules in a python data file (*.npy)
# The input data (synaptic activities) and the output (BOLD time-series) are numpy arrays
# with columns in the following order:
#
# V1 ROI (right hemisphere, includes LSNM units and TVB nodes) 
# V4 ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (right hemisphere, includes LSNM units and TVB nodes)
# FS ROI (right hemisphere, includes LSNM units and TVB nodes)
# D1 ROI (right hemisphere, includes LSNM units and TVB nodes)
# D2 ROI (right hemisphere, includes LSNM units and TVB nodes)
# FR ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (left hemisphere, contains only  TVB nodes)


import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_ROI.npy'

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'lsnm_bold_poisson.npy'

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 5 ms.
Ti = 0.005 * 10

# define constant needed for hemodynamic function (in milliseconds)
lambda_ = 6

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# Time for one complete trial in seconds
Ttrial = 5.5

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

# how many scans do you want to remove from beginning of BOLD timeseries?
scans_to_remove = 8

# read the input file that contains the synaptic activities of all ROIs
syn = np.load(SYN_file)

# extract the synaptic activities corresponding to each ROI:
v1_syn = syn[0]
v4_syn = syn[1]
it_syn = syn[2]
fs_syn = syn[3]
d1_syn = syn[4]
d2_syn = syn[5]
fr_syn = syn[6]
lit_syn= syn[7]

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = v1_syn.size

# Given neural synaptic time interval and total time of scanning experiment,
# construct a numpy array of time points (data points provided in data files)
time_in_seconds = np.arange(0, T, Tr)

# the following calculates a Poisson distribution (that will represent a hemodynamic
# function, given lambda (the Poisson time constant characterizing width and height
# of hemodynamic function), and tau (the time step)
# if you would do it manually you would do the following:
#h = [lambda_ ** tau * m.exp(-lambda_) / m.factorial(tau) for tau in time_in_seconds]
h = poisson.pmf(time_in_seconds, lambda_)

# the following calculates the impulse response (convolution kernel) of the gamma
# function, approximating the BOLD response (Boynton model).

# rescale the array containing the poisson to increase its size and match the size of
# the synaptic activity array (using linear interpolation)
scanning_timescale = np.arange(0, synaptic_timesteps, synaptic_timesteps / (T/Tr))
synaptic_timescale = np.arange(0, synaptic_timesteps)

h = np.interp(synaptic_timescale, scanning_timescale, h)

# now, we need to convolve the synaptic activity with a hemodynamic delay
# function and sample the array at Tr regular intervals (back to the scanning
# timescale)
v1_BOLD = np.convolve(v1_syn, h)[scanning_timescale]
v4_BOLD = np.convolve(v4_syn, h)[scanning_timescale]
it_BOLD = np.convolve(it_syn, h)[scanning_timescale]
d1_BOLD = np.convolve(d1_syn, h)[scanning_timescale]
d2_BOLD = np.convolve(d2_syn, h)[scanning_timescale]
fs_BOLD = np.convolve(fs_syn, h)[scanning_timescale]
fr_BOLD = np.convolve(fr_syn, h)[scanning_timescale]
lit_BOLD= np.convolve(lit_syn,h)[scanning_timescale]

# now we are going to remove the first trial
# estimate how many 'synaptic ticks' there are in each trial
synaptic_ticks = Ttrial/Ti
# estimate how many 'MR ticks' there are in each trial
mr_ticks = round(Ttrial/Tr)

# remove first few scans from BOLD signal array (to eliminate edge effects from
# convolution)
v1_BOLD = np.delete(v1_BOLD, np.arange(scans_to_remove))
v4_BOLD = np.delete(v4_BOLD, np.arange(scans_to_remove))
it_BOLD = np.delete(it_BOLD, np.arange(scans_to_remove))
d1_BOLD = np.delete(d1_BOLD, np.arange(scans_to_remove))
d2_BOLD = np.delete(d2_BOLD, np.arange(scans_to_remove))
fs_BOLD = np.delete(fs_BOLD, np.arange(scans_to_remove))
fr_BOLD = np.delete(fr_BOLD, np.arange(scans_to_remove))
lit_BOLD= np.delete(lit_BOLD,np.arange(scans_to_remove))

# ...and normalize the BOLD signal of each module (convert to percentage signal change)
#v1_BOLD = v1_BOLD / np.mean(v1_BOLD) * 100. - 100.
#v4_BOLD = v4_BOLD / np.mean(v4_BOLD) * 100. - 100.
#it_BOLD = it_BOLD / np.mean(it_BOLD) * 100. - 100.
#d1_BOLD = d1_BOLD / np.mean(d1_BOLD) * 100. - 100.
#d2_BOLD = d2_BOLD / np.mean(d2_BOLD) * 100. - 100.
#fs_BOLD = fs_BOLD / np.mean(fs_BOLD) * 100. - 100.
#fr_BOLD = fr_BOLD / np.mean(fr_BOLD) * 100. - 100.

# pack BOLD time-series in preparation for saving to output file
lsnm_BOLD = np.array([v1_BOLD, v4_BOLD, it_BOLD,
                      fs_BOLD, d1_BOLD, d2_BOLD, fr_BOLD,
                      lit_BOLD ])

print lsnm_BOLD.shape

# now, save all BOLD timeseries to a single file 
# Please note that we are saving the original BOLD time-series, before removing the
# edge effects
np.save(BOLD_file, lsnm_BOLD)

# Set up figure to plot synaptic activity
plt.figure(1)

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot synaptic activities
plt.plot(v1_syn)
plt.plot(it_syn)
plt.plot(d1_syn)

# Set up separate figures to plot fMRI BOLD signal
plt.figure(2)

plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN V1/V2')

plt.plot(v1_BOLD, linewidth=3.0, color='yellow')
plt.gca().set_axis_bgcolor('black')

print v1_BOLD.shape

plt.figure(4)

plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN V4')

plt.plot(v4_BOLD, linewidth=3.0, color='green')
plt.gca().set_axis_bgcolor('black')

plt.figure(5)
plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN IT')

plt.plot(it_BOLD, linewidth=3.0, color='blue')
plt.gca().set_axis_bgcolor('black')

plt.figure(6)
plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN D1')

plt.plot(d1_BOLD, linewidth=3.0, color='red')
plt.gca().set_axis_bgcolor('black')

plt.figure(7)
plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN LEFT IT')

plt.plot(lit_BOLD, linewidth=3.0, color='pink')
plt.gca().set_axis_bgcolor('black')

# Show the plots on the screen
plt.show()
