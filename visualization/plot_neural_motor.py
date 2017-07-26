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
#   This file (plot_neural_motor.py) was created on July 20, 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on July 23 2017  
# **************************************************************************/

# plot_neural_motor.py
#
# Plot output data files of motor simulation

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# Load data files
eml23 = np.loadtxt('eml23.out')
im    = np.loadtxt('im.out')
eml5  = np.loadtxt('eml5.out')

avg_m = np.mean(eml5, axis=1)

print 'Number of timesteps: ', avg_m.shape[0]
print 'Number of neuronal units: ', eml5.shape[1]

time=np.linspace(0, int(avg_m.shape[0]*0.1), 5)

print 'Time array: ', time


# Set up plot to display average neural of M1 Layer 5 activity 
fig1 = plt.figure('Average M1 Layer 5 Pyramidal neuronal activity')
ax= fig1.add_subplot(111)

# increase font size
#plt.rcParams.update({'font.size': 15})

# Plot M1 module and TMS pulse
ax.plot(avg_m, color='r')
ax.set_ylim(0, 0.3)
ax.axvline(x=30, color='black')

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)
axarr[0].set_title('Neural population activity')
axarr[0].imshow(eml23.T, vmin=0., vmax=0.6, interpolation='nearest', cmap='hot')
axarr[0].axvline(x=30, color='black')
axarr[0].set_yticklabels([])
axarr[0].set_ylabel('L2/3 (E)')
axarr[1].imshow(im.T, vmin=0., vmax=0.6, interpolation='nearest', cmap='hot')
axarr[1].axvline(x=30, color='black')
axarr[1].set_yticklabels([])
axarr[1].set_ylabel('(I)')
axarr[2].imshow(eml5.T, vmin=0., vmax=0.6, interpolation='nearest', cmap='hot')
axarr[2].axvline(x=30, color='black')
axarr[2].set_yticklabels([])
axarr[2].set_ylabel('L5 (E)')


# Set up plot to display neural population activity of each unit in Layer 2/3 (heatmap)
fig = plt.figure('Neural excitatory activity across M1 Layer 2/3 Pyramidal neurons')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(eml23.T,
                vmin=0., vmax=0.6,
                interpolation='nearest', cmap='hot')
plt.axvline(x=30, color='black')
ax.grid(False)
color_bar=plt.colorbar(cax, orientation='horizontal')

# Set up plot to display neural population activity of each unit in Layer 5 (heatmap)
fig = plt.figure('Neural excitatory activity across M1 Layer 5 Pyramidal neurons')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(eml5.T,
                vmin=0., vmax=0.6,
                interpolation='nearest', cmap='hot')
plt.axvline(x=30, color='black')
# Set up plot to display neural population activity of each unit in Layer 2/3 (heatmap)
color_bar=plt.colorbar(cax, orientation='horizontal')

fig = plt.figure('Neural inhibitory activity')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(im.T,
                vmin=-0., vmax=0.6,
                interpolation='nearest', cmap='hot')
plt.axvline(x=30, color='black')
ax.grid(False)
color_bar=plt.colorbar(cax, orientation='horizontal')

# Show the plot on the screen
plt.show()

