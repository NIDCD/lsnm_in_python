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
#   This file (plot_synaptic_visual.py) was created on April 17, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on July 19 2015  
# **************************************************************************/

# plot_synaptic_visual.py
#
# Plot synaptic activity data from visual
# delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

# what are the locations of relevant TVB nodes within TVB array?
#v1_loc = 345
#v4_loc = 393
#it_loc = 413
#pf_loc =  74

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_synaptic.npy")

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the excel document:
#       "Hagmann's Talairach Coordinates (obtained from TVB).xlsx"
# Extracted from The Virtual Brain Demo Data Sets
# Please note that arrays in Python start from zero so one does need to account for that and shift
# indices given by the above document by one location.
# Use 6 nodes within rPCAL
v1_loc = range(344, 350)     # Hagmann's brain nodes included within V1 ROI

# Use 6 nodes within rFUS
v4_loc = range(390, 396)     # Hagmann's brain nodes included within V4 ROI       

# Use 6 nodes within rPARH
it_loc = range(412, 418)     # Hagmann's brain nodes included within IT ROI

# Use 6 nodes within rRMF
d1_loc = range(73, 79)       # Hagmann's brain nodes included within D1 ROI

# Use 6 nodes within rPTRI
d2_loc = range(39, 45)       # Hagmann's brain nodes included within D2 ROI

# Use 6 nodes within rPOPE
fs_loc = range(47, 53)       # Hagmann's brain nodes included within FS ROI

# Use 6 nodes within rCMF
fr_loc = range(125, 131)     # Hagmann's brain nodes included within FR ROI


# Load TVB host node synaptic activities into separate numpy arrays
# the index '0' stores excitary (E) synaptic activity, and index '1'
# stores inhibitory synaptic activity
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

# Load V1 synaptic activity data files into a numpy array
ev1h = np.loadtxt('ev1h_synaptic.out')
ev1v = np.loadtxt('ev1v_synaptic.out')
iv1h = np.loadtxt('iv1h_synaptic.out')
iv1v = np.loadtxt('iv1v_synaptic.out')

# Load IT synaptic activity data files into a numpy array
exss = np.loadtxt('exss_synaptic.out')
inss = np.loadtxt('inss_synaptic.out')

# Load D1 synaptic activity data files into a numpy array
efd1 = np.loadtxt('efd1_synaptic.out')
ifd1 = np.loadtxt('ifd1_synaptic.out')

# Extract number of timesteps from one of the matrices
timesteps = ev1h.shape[0]
print timesteps

# the following variable defines the timesteps we will see in the resulting plot
# we also convert the number of timesteps to seconds by multiplying by 50 and dividng by 1000
ts_to_plot = 660
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
v1 = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1) + np.sum(tvb_ev1 + tvb_iv1, axis=1)
it = np.sum(exss + inss, axis = 1) + np.sum(tvb_eit + tvb_iit, axis=1)
d1 = np.sum(efd1 + ifd1, axis = 1) + np.sum(tvb_ed1 + tvb_id1, axis=1)

# Set up plot
plt.figure(1)

#plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# increase font size
plt.rcParams.update({'font.size': 30})

ax1=plt.subplot()

ax1.set_ylim([600, 2100])
ax1.set_xlim(0,32.95)

# Plot V1 module
plt.plot(t, v1[0:ts_to_plot], color='yellow', linewidth=2)
plt.plot(t, d1[0:ts_to_plot], color='red', linewidth=2)
plt.plot(t, it[0:ts_to_plot], color='blue', linewidth=2)

plt.gca().set_axis_bgcolor('black')

plt.xlabel('Time (s)')

plt.tight_layout()

# Show the plot on the screen
plt.show()

