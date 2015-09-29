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
#   This file (plot_neural_tvb.py) was created on September 13, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on September 13, 2015  
# **************************************************************************/

# plot_neural_tvb.py
#
# Plot output data files of Hagmann's brain simulation using TVB

# what are the locations of relevant TVB nodes within TVB array?
v1_loc = 344
v4_loc = 390
it_loc = 423
fs_loc =  41
d1_loc =  74
d2_loc =  47
fr_loc = 125

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# load file containing TVB nodes electrical activity
tvb = np.load('tvb_neuronal.npy')

tvb_v1 = tvb[:, 0, v1_loc, 0]
tvb_v4 = tvb[:, 0, v4_loc, 0]
tvb_it = tvb[:, 0, it_loc, 0]
tvb_fs = tvb[:, 0, fs_loc, 0]
tvb_d1 = tvb[:, 0, d1_loc, 0]
tvb_d2 = tvb[:, 0, d2_loc, 0]
tvb_fr = tvb[:, 0, fr_loc, 0]

# Extract number of timesteps from one of the matrices
timesteps = tvb_v1.shape[0]

# what was the duration of simulation in real time (in ms)?
real_duration = 16500

print timesteps

# Contruct a numpy array of timesteps (data points provided in data file)
real_time = np.linspace(0, real_duration, num=timesteps)

# Set up plot
fig1=plt.figure(1, facecolor='white')

# Plot V1 module
ax = plt.subplot(7,1,1)
ax.plot(real_time, tvb_v1, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
ax.set_title('SIMULATED ELECTRICAL ACTIVITY, HAGMANNS BRAIN')
plt.ylabel('V1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,2)
ax.plot(real_time, tvb_v4, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
plt.ylabel('V4', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,3)
ax.plot(real_time, tvb_it, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,4)
ax.plot(real_time, tvb_fs, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,5)
ax.plot(real_time, tvb_d1, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,6)
ax.plot(real_time, tvb_d2, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,7)
ax.plot(real_time, tvb_fr, color='b', linewidth=2)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,real_duration)
plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')

plt.tight_layout()

# Show the plot on the screen
plt.show()

