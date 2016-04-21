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
#   This file (compute_syn_auditory.py) was created on April 1, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on April 11 2016
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_syn_auditory.py
#
# Calculate and plot simulated synaptic activities in given ROIs 
# 
# ... using data from visual delay-match-to-sample simulation.
# It also saves the synaptic activities for each and all modules in a python data file
# (*.npy)
# The data is saved in a numpy array with columns in the following order:
#
# A1 ROI (right hemisphere, includes LSNM units and TVB nodes) 
# A2 ROI (right hemisphere, includes LSNM units and TVB nodes)
# ST ROI (right hemisphere, includes LSNM units and TVB nodes)
# PF ROI (right hemisphere, includes LSNM units and TVB nodes)

import numpy as np

import matplotlib.pyplot as plt

# define the name of the output file where the integrated synaptic activity will be stored
syn_file = 'synaptic_in_ROI.npy'

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the excel document:
#       "Location of Auditory LSNM modules within Connectome.xlsx"
# Extracted from The Virtual Brain Demo Data Sets
# Please note that arrays in Python start from zero and so do the
# indices given by the above document.
a1_loc = [474, 473, 498, 472, 466]     # Hagmann's brain nodes included within A1 ROI

a2_loc = [470, 471, 465, 469, 476]     # Hagmann's brain nodes included within A2 ROI       

st_loc = [477, 475, 478, 497, 482]     # Hagmann's brain nodes included within ST ROI

pf_loc = [ 51, 140, 143,  50, 139]     # Hagmann's brain nodes included within PFC ROI

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_abs_syn.npy")

# Load TVB host node synaptic activities into separate numpy arrays
tvb_ea1 = tvb_synaptic[:, 0, a1_loc[0]:a1_loc[-1]+1, 0]
tvb_ea2 = tvb_synaptic[:, 0, a2_loc[0]:a2_loc[-1]+1, 0]
tvb_est = tvb_synaptic[:, 0, st_loc[0]:st_loc[-1]+1, 0]
tvb_epf = tvb_synaptic[:, 0, pf_loc[0]:pf_loc[-1]+1, 0]

tvb_ia1 = tvb_synaptic[:, 1, a1_loc[0]:a1_loc[-1]+1, 0]
tvb_ia2 = tvb_synaptic[:, 1, a2_loc[0]:a2_loc[-1]+1, 0]
tvb_ist = tvb_synaptic[:, 1, st_loc[0]:st_loc[-1]+1, 0]
tvb_ipf = tvb_synaptic[:, 1, pf_loc[0]:pf_loc[-1]+1, 0]

# Load LSNM synaptic activity data files into a numpy arrays
ea1u = np.loadtxt('ea1u_abs_syn.out')
ea1d = np.loadtxt('ea1d_abs_syn.out')
ia1u = np.loadtxt('ia1u_abs_syn.out')
ia1d = np.loadtxt('ia1d_abs_syn.out')
ea2u = np.loadtxt('ea2u_abs_syn.out')
ea2c = np.loadtxt('ea2c_abs_syn.out')
ea2d = np.loadtxt('ea2d_abs_syn.out')
ia2u = np.loadtxt('ia2u_abs_syn.out')
ia2c = np.loadtxt('ia2c_abs_syn.out')
ia2d = np.loadtxt('ia2d_abs_syn.out')
estg = np.loadtxt('estg_abs_syn.out')
istg = np.loadtxt('istg_abs_syn.out')
efd1 = np.loadtxt('efd1_abs_syn.out')
ifd1 = np.loadtxt('ifd1_abs_syn.out')
efd2 = np.loadtxt('efd2_abs_syn.out')
ifd2 = np.loadtxt('ifd2_abs_syn.out')
exfs = np.loadtxt('exfs_abs_syn.out')
infs = np.loadtxt('infs_abs_syn.out')
exfr = np.loadtxt('exfr_abs_syn.out')
infr = np.loadtxt('infr_abs_syn.out')

# add all units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
a1_syn = np.sum(ea1u + ea1d + ia1u + ia1d, axis = 1) #+ np.sum(tvb_ea1+tvb_ia1, axis=1)
a2_syn = np.sum(ea2u + ea2c + ea2d + ia2u + ia2c + ia2d, axis = 1) #+ np.sum(tvb_ea2+tvb_ia2, axis=1)
st_syn = np.sum(estg + istg, axis = 1) #+ np.sum(tvb_est+tvb_ist, axis=1)
d1_syn = np.sum(efd1 + ifd1, axis = 1)
d2_syn = np.sum(efd2 + ifd2, axis = 1)
fs_syn = np.sum(exfs + infs, axis = 1)
fr_syn = np.sum(exfr + infr, axis = 1)

pf_syn = d1_syn + d2_syn + fs_syn + fr_syn #+ np.sum(tvb_epf+tvb_ipf, axis=1)

# get rid of the first time point ('zero point') bc it could skew correlations later
#v1_syn[0] = v1_syn[1]
#v4_syn[0] = v4_syn[1]
#it_syn[0] = it_syn[1]
#d1_syn[0] = d1_syn[1]
#d2_syn[0] = d2_syn[1]
#fs_syn[0] = fs_syn[1]
#fr_syn[0] = fr_syn[1]
#lit_syn[0] = lit_syn[1]

# create a numpy array of timeseries
synaptic = np.array([a1_syn, a2_syn, st_syn, pf_syn])

# now, save all synaptic timeseries to a single file 
np.save(syn_file, synaptic)

# Extract number of timesteps from one of the matrices
timesteps = a1_syn.shape[0]
print 'Timesteps = ', timesteps

# Construct a numpy array of timesteps (data points provided in data file)
# to convert from timesteps to time in seconds we do the following:
# Each simulation time-step equals 5 milliseconds
# However, we are recording only once every 10 time-steps
# Therefore, each data point in the output files represents 50 milliseconds.
# Thus, we need to multiply the datapoint times 50 ms...
# ... and divide by 1000 to convert to seconds
#t = np.linspace(0, 659*50./1000., num=660)
t = np.linspace(0, timesteps * 35.0 / 1000., num=timesteps)


# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')
plt.plot(t, a1_syn, label='A1')
plt.plot(t, a2_syn, label='A2')
plt.plot(t, st_syn, label='ST')
plt.plot(t, pf_syn, label='PFC')

plt.legend()

# Show the plots on the screen
plt.show()
