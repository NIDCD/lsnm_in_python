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
#   This file (compute_meg_auditory.py) was created on June 7, 2015.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on March 15 2016 
# **************************************************************************/

# compute_meg_auditory.py
#
# Calculate and plot MEG signal at source locations based on data from auditory
# delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

# define the name of the output file where the MEG source activity timeseries will be stored
MEG_source_file = 'meg_source_activity.npy'

# Load A1 synaptic activity data files into a numpy array
ea1u = np.loadtxt('ea1u_signed_syn.out')
ea1d = np.loadtxt('ea1d_signed_syn.out')

# Load A2 synaptic activity data files into a numpy array
ea2u = np.loadtxt('ea2u_signed_syn.out')
ea2c = np.loadtxt('ea2c_signed_syn.out')
ea2d = np.loadtxt('ea2d_signed_syn.out')

# Load ST synaptic activity data files into a numpy array
estg = np.loadtxt('estg_signed_syn.out')

# Load PFC synaptic activity data files into a numpy array
efd1 = np.loadtxt('efd1_signed_syn.out')
efd2 = np.loadtxt('efd2_signed_syn.out')
exfs = np.loadtxt('exfs_signed_syn.out')
exfr = np.loadtxt('exfr_signed_syn.out')

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = ea1u.shape[0]

# add all units within each region together across space to calculate
# MEG source dynamics in each brain region
a1 = np.sum(ea1u + ea1d, axis = 1)
a2 = np.sum(ea2u + ea2c + ea2d, axis=1)
st = np.sum(estg, axis = 1)
pf = np.sum(efd1 + efd2 + exfs + exfr, axis = 1)

# create a numpy array of MEG source activity timeseries
meg_source = np.array([a1, a2, st, pf])

print 'Size of each MEG source activity time-series: ', a1.size

# now, save all MEG source activity timeseries to a single file 
np.save(MEG_source_file, meg_source)

# Set up figure to plot MEG source dynamics
plt.figure(1)

plt.suptitle('SIMULATED MEG SOURCE DYNAMICS')

# Plot MEG signal
a1_plot=plt.plot(a1, label='A1')
a2_plot=plt.plot(a2, label='A2')
st_plot=plt.plot(st, label='ST')
pf_plot=plt.plot(pf, label='PFC')

plt.legend()

# Show the plot on the screen
plt.show()
