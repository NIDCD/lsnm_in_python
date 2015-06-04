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
#   This file (plotSynapticAuditory.py) was created on April 17, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on April 17 2015  
# **************************************************************************/

# plotSynapticAuditory.py
#
# Plot synaptic activity data from auditory delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

# Load V1 synaptic activity data files into a numpy array
ea1u = np.loadtxt('../../output/ea1u_synaptic.out')
ea1d = np.loadtxt('../../output/ea1d_synaptic.out')
ia1u = np.loadtxt('../../output/ia1u_synaptic.out')
ia1d = np.loadtxt('../../output/ia1d_synaptic.out')

# Load IT synaptic activity data files into a numpy array
estg = np.loadtxt('../../output/estg_synaptic.out')
istg = np.loadtxt('../../output/istg_synaptic.out')

# Load D1 synaptic activity data files into a numpy array
efd1 = np.loadtxt('../../output/efd1_synaptic.out')
ifd1 = np.loadtxt('../../output/ifd1_synaptic.out')


# Extract number of timesteps from one of the matrices
timesteps = ea1u.shape[0]

# Contruct a numpy array of timesteps (data points provided in data files)
t = np.arange(0, timesteps, 1)

# add all units within each region (V1, IT, and D1) together across space
a1 = np.sum(ea1u + ea1d + ia1u + ia1d, axis = 1)
st = np.sum(estg + istg, axis = 1)
d1 = np.sum(efd1 + ifd1, axis = 1)

# Set up plot
plt.figure(1)

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot V1 module
plt.plot(t, a1)
plt.plot(t, st)
plt.plot(t, d1)

# Show the plot on the screen
plt.show()

