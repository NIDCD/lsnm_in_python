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
#   This file (compute_syn_visual_66_regions.py) was created on October 11, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on October 11, 2016
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_syn_visual_66_regions.py
#
# Calculate and plot simulated synaptic activities in 33 ROIs (right hemisphere)
# as defined by Haggman et al
# 
# ... using data from visual delay-match-to-sample simulation (or resting state simulation
# of the same duration as the DMS).
# It also saves the synaptic activities for 33 ROIs (right hemisphere) in a python data file
# (*.npy)
# The data is saved in a numpy array where the columns are the 33 ROIs:
#

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# define the name of the output file where the integrated synaptic activity will be stored
syn_file = 'synaptic_in_66_ROIs.npy'

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the excel document:
#       "Hagmann's Talairach Coordinates (obtained from TVB).xlsx"
# Extracted from The Virtual Brain Demo Data Sets
# Please note that arrays in Python start from zero so one does need to account for that and shift
# indices given by the above document by one location.
# Use 6 nodes within rPCAL including host node 345
v1_loc = range(344, 350)     # Hagmann's brain nodes included within V1 ROI

# Use 6 nodes within rFUS including host node 393
v4_loc = range(390, 396)     # Hagmann's brain nodes included within V4 ROI       

# Use 6 nodes within rPARH including host node 413
it_loc = range(412, 418)     # Hagmann's brain nodes included within IT ROI

# Use 6 nodes within rRMF including host node 74
d1_loc = range(73, 79)       # Hagmann's brain nodes included within D1 ROI

# Use 6 nodes within rPTRI including host node 41
d2_loc = range(39, 45)       # Hagmann's brain nodes included within D2 ROI

# Use 6 nodes within rPOPE including host node 47
fs_loc = range(47, 53)       # Hagmann's brain nodes included within FS ROI

# Use 6 nodes within rCMF including host node 125
fr_loc = range(125, 131)     # Hagmann's brain nodes included within FR ROI

# Use 6 nodes within lPARH
lit_loc= range(911, 917)     # Hagmann's brain nodes included within left IT ROI

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_abs_syn.npy")

# Load TVB host node synaptic activities into separate numpy arrays
tvb_ev1 = tvb_synaptic[:, 0, v1_loc[0]:v1_loc[-1]+1, 0]
tvb_ev4 = tvb_synaptic[:, 0, v4_loc[0]:v4_loc[-1]+1, 0]
tvb_eit = tvb_synaptic[:, 0, it_loc[0]:it_loc[-1]+1, 0]
tvb_ed1 = tvb_synaptic[:, 0, d1_loc[0]:d1_loc[-1]+1, 0]
tvb_ed2 = tvb_synaptic[:, 0, d2_loc[0]:d2_loc[-1]+1, 0]
tvb_efs = tvb_synaptic[:, 0, fs_loc[0]:fs_loc[-1]+1, 0]
tvb_efr = tvb_synaptic[:, 0, fr_loc[0]:fr_loc[-1]+1, 0]
tvb_iv1 = tvb_synaptic[:, 1, v1_loc[0]:v1_loc[-1]+1, 0]
tvb_iv4 = tvb_synaptic[:, 1, v4_loc[0]:v4_loc[-1]+1, 0]
tvb_iit = tvb_synaptic[:, 1, it_loc[0]:it_loc[-1]+1, 0]
tvb_id1 = tvb_synaptic[:, 1, d1_loc[0]:d1_loc[-1]+1, 0]
tvb_id2 = tvb_synaptic[:, 1, d2_loc[0]:d2_loc[-1]+1, 0]
tvb_ifs = tvb_synaptic[:, 1, fs_loc[0]:fs_loc[-1]+1, 0]
tvb_ifr = tvb_synaptic[:, 1, fr_loc[0]:fr_loc[-1]+1, 0]

# now extract synaptic activity in the contralateral IT
tvb_elit = tvb_synaptic[:, 0, lit_loc[0]:lit_loc[-1]+1, 0]
tvb_ilit = tvb_synaptic[:, 1, lit_loc[0]:lit_loc[-1]+1, 0]

# add all units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
v1_syn = np.sum(tvb_ev1+tvb_iv1, axis=1)
v4_syn = np.sum(tvb_ev4+tvb_iv4, axis=1)
it_syn = np.sum(tvb_eit+tvb_iit, axis=1)
d1_syn = np.sum(tvb_ed1+tvb_id1, axis=1)
d2_syn = np.sum(tvb_ed2+tvb_id2, axis=1)
fs_syn = np.sum(tvb_efs+tvb_ifs, axis=1)
fr_syn = np.sum(tvb_efr+tvb_ifr, axis=1)

# now, add unit across space in the contralateral IT
lit_syn = np.sum(tvb_elit + tvb_ilit, axis=1)

# create a numpy array of timeseries
synaptic = np.array([v1_syn, v4_syn, it_syn, fs_syn, d1_syn, d2_syn, fr_syn, lit_syn])

# now, save all synaptic timeseries to a single file 
np.save(syn_file, synaptic)

# Extract number of timesteps from one of the matrices
timesteps = v1_syn.shape[0]
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
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN V1')
plt.plot(t, v1_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN V4')
plt.plot(v4_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN IT')
plt.plot(it_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN FS')
plt.plot(fs_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN D1')
plt.plot(d1_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN D2')
plt.plot(d2_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN FR')
plt.plot(fr_syn)
# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY IN LEFT IT')
plt.plot(lit_syn)

# Show the plots on the screen
plt.show()
