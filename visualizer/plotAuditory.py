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
#   This file (plotAuditory.py) was created on March 25, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on March 25, 2015  
# **************************************************************************/

# plotAuditory.py
#
# Plot output data files of auditory delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

# Load data files
mgns = np.loadtxt('../../output/mgns.out')
efd1 = np.loadtxt('../../output/efd1.out')
efd2 = np.loadtxt('../../output/efd2.out')
ea1d = np.loadtxt('../../output/ea1d.out')
ea1u = np.loadtxt('../../output/ea1u.out')
ea2d = np.loadtxt('../../output/ea2d.out')
ea2u = np.loadtxt('../../output/ea2u.out')
ea2c = np.loadtxt('../../output/ea2c.out')
exfr = np.loadtxt('../../output/exfr.out')
exfs = np.loadtxt('../../output/exfs.out')
estg = np.loadtxt('../../output/estg.out')

# Extract number of timesteps from one of the matrices
timesteps = mgns.shape[0]

# Contruct a numpy array of timesteps (data points provided in data file)
t = np.arange(0, timesteps, 1)

# Set up plot
plt.figure(1)

plt.suptitle('SIMULATED NEURAL ACTIVITY')

# Plot MGN module
ax = plt.subplot(11,1,1)
ax.plot(t, mgns)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('MGN', rotation='horizontal', horizontalalignment='right')

# Plot A1 module
ax = plt.subplot(11,1,2)
ax.plot(t, ea1d)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('A1d', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(11,1,3)
ax.plot(t, ea1u)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('A1u', rotation='horizontal', horizontalalignment='right')

# Plot A2 module
ax = plt.subplot(11,1,4)
ax.plot(t, ea2d)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('A2d', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(11,1,5)
ax.plot(t, ea2u)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('A2u', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(11,1,6)
ax.plot(t, ea2c)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('A2c', rotation='horizontal', horizontalalignment='right')

# Plot IT module
ax = plt.subplot(11,1,7)
ax.plot(t, estg)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('STG', rotation='horizontal', horizontalalignment='right')

# Plot PFC modules FS, FD1, and FD2
ax = plt.subplot(11,1,8)
ax.plot(t, exfs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(11,1,9)
ax.plot(t, efd1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(11,1,10)
ax.plot(t, efd2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

# Plot FR (Response module)
ax = plt.subplot(11,1,11)
ax.plot(t, exfr)
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('R', rotation='horizontal', horizontalalignment='right')
plt.xlabel('Timesteps (i.e., Data points)')

# Show the plot on the screen
plt.show()

