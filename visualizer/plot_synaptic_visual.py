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

# what are the locations of relevant TVB nodes within TVB array?
v1_loc = 345
v4_loc = 393
it_loc = 413
pf_loc =  74

import numpy as np
import matplotlib.pyplot as plt

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_synaptic.npy")

# Load V1 synaptic activity data files into a numpy array
ev1h = np.loadtxt('ev1h_synaptic.out')
ev1v = np.loadtxt('ev1v_synaptic.out')
iv1h = np.loadtxt('iv1h_synaptic.out')
iv1v = np.loadtxt('iv1v_synaptic.out')

# Load TVB V1 host node synaptic activity into numpy array
tvb_v1 = tvb_synaptic[:, v1_loc]

# Load IT synaptic activity data files into a numpy array
exss = np.loadtxt('exss_synaptic.out')
inss = np.loadtxt('inss_synaptic.out')

# Load TVB IT host node synaptic activity into numpy array
tvb_it = tvb_synaptic[:, it_loc]

# Load D1 synaptic activity data files into a numpy array
efd1 = np.loadtxt('efd1_synaptic.out')
ifd1 = np.loadtxt('ifd1_synaptic.out')

# Load TVB D1 host node synaptic activity into numpy array
tvb_d1 = tvb_synaptic[:, pf_loc]

# Extract number of timesteps from one of the matrices
timesteps = ev1h.shape[0]

# Construct a numpy array of timesteps (data points provided in data files)
t = np.arange(0, timesteps, 1)

# add all units within each region (V1, IT, and D1) together across space
v1 = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1) + tvb_v1
it = np.sum(exss + inss, axis = 1) + tvb_it
d1 = np.sum(efd1 + ifd1, axis = 1) + tvb_d1

# Set up plot
plt.figure(1)

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot V1 module
plt.plot(t, v1)
plt.plot(t, it)
plt.plot(t, d1)

# Show the plot on the screen
plt.show()

