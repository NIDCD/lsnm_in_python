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
#   This file (compute_syn_visual_998_regions.py) was created on March 8 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on March 8 2017
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_syn_visual_998_regions.py
#
# Calculate and plot simulated synaptic activities in 998 ROIs
# as defined by Haggman et al
# 
# ... using data from visual delay-match-to-sample simulation (or resting state simulation
# of the same duration as the DMS).
# It also saves the synaptic activities for the 998 ROIs in a python data file
# (*.npy)
# The data is saved in a numpy array where the columns are the 998 ROIs' integrated synaptic
# activity


import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

# define total number of ROIs
ROIs = 998

# define the name of the output file where the integrated synaptic activity will be stored
syn_file = 'synaptic_in_998_ROIs.npy'

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the excel document:
#       "Location of visual LSNM modules within Connectome.xlsx"
# Extracted from The Virtual Brain Demo Data Sets

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_abs_syn.npy")

print 'Shape of synaptic array: ', tvb_synaptic.shape

# create a numpy array of synaptic time-series, with a number of elements defined
# by the number of ROIs above and the number of time points in each synaptic time-series
synaptic = np.empty([ROIs, tvb_synaptic.shape[0]])

print 'Shape of output integrated synaptic activity array: ', synaptic.shape

# Load TVB host node synaptic activities into separate numpy arrays
# add excitatory and inhibitory values of each ROI's synaptic activity
for idx in range(0, ROIs):
    synaptic[idx] = tvb_synaptic[:, 0, idx, 0] + tvb_synaptic[:, 1, idx, 0]  
                                                    
# now, save all synaptic timeseries to a single file 
np.save(syn_file, synaptic)

# Extract total number of timesteps from synaptic time-series
timesteps = tvb_synaptic.shape[0]
print 'Timesteps = ', timesteps

# Construct a numpy array of timesteps (data points provided in data file)
# to convert from timesteps to time in seconds we do the following:
# Each simulation time-step equals 5 milliseconds
# However, we are recording only once every 10 time-steps
# Therefore, each data point in the output files represents 50 milliseconds.
# Thus, we need to multiply the datapoint times 50 ms...
# ... and divide by 1000 to convert to seconds
#t = np.linspace(0, 659*50./1000., num=660)
t = np.linspace(0, timesteps * 50.0 / 1000., num=timesteps)


# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY OF ONE ROI')
plt.plot(t, synaptic[0])

# Show the plots on the screen
plt.show()
