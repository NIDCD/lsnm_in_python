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
#   This file (compute_FC_syn_vis.py) was created on July 20 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on July 20 2017
#
# **************************************************************************/

# compute_FC_syn_vis.py
#
# Compute, display, and save functional connectivity matrix using given synaptic timeseries

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm as CM


# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_998_ROIs.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
xcorr_file = 'xcorr_matrix_998_regions_syn.npy'

# read the input file that contains the synaptic activities of all ROIs
syn = np.load(SYN_file)

# claculate total number of ROIs
ROIs = syn.shape[0]

print 'Dimensions of synaptic file: ', syn.shape
print 'LENGTH OF SYNAPTIC TIME-SERIES: ', syn[0].size
print 'Number of ROIs: ', ROIs

# First value of each synaptic array is always zero. So make it equal to second value
syn[:, 0] = syn[:, 1]

# normalize synaptic activities (needed prior to BOLD estimation)
#for idx, roi in enumerate(syn):
#    syn[idx] = (syn[idx] - syn[idx].min()) / (syn[idx].max() - syn[idx].min())

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = syn[0].size
print 'Size of synaptic arrays: ', synaptic_timesteps

plt.figure()

plt.suptitle('SIMULATED SYNAPTIC SIGNAL')

# plot synaptic time-series for all ROIs
for idx, roi in enumerate(syn):
    plt.plot(syn[idx], linewidth=1.0, color='black')

# calculate correlation matrix
corr_mat = np.corrcoef(syn)
print 'Number of ROIs: ', syn.shape
print 'Shape of correlation matrix: ', corr_mat.shape

# save cross-correlation matrix to an npy python file
np.save(xcorr_file, corr_mat)

#initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(corr_mat, interpolation='nearest', cmap=cmap)
ax.grid(False)
plt.colorbar(cax)

# Show the plots on the screen
plt.show()
