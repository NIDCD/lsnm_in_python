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
#   This file (compute_time_resolved_FC.py) was created on November 29 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on November 29 2017
#
# **************************************************************************/

# compute_time_resolved_FC.py
#
# Given a 998-node fMRI BOLD timeseries, calculate and save time-resolved FC
# matrices

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint

from scipy.stats import kurtosis

from scipy.stats import skew

from matplotlib import cm as CM

from scipy.signal import butter, lfilter, freqz

# define the name of the input file where the BOLD timeseries will is stored
BOLD_file = 'bold_balloon_998_regions_3T_0.25Hz.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
corr_file = 'time_resolved_FC_matrix_998_regions_3T_0.25Hz.npy'

fMRI_BOLD = np.load(BOLD_file)

# calculate correlation matrix
corr_mat = np.corrcoef(fMRI_BOLD)
print 'Number of ROIs: ', fMRI_BOLD.shape
print 'Shape of correlation matrix: ', corr_mat.shape

# save cross-correlation matrix to an npy python file
np.save(corr_file, corr_mat)

#initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(corr_mat, interpolation='nearest', cmap=cmap)
ax.grid(False)
plt.colorbar(cax)

# initialize new figure for plotting histogram
fig = plt.figure()
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(corr_mat.shape[1], k=0)
mask = np.transpose(mask)

# apply mask to empirical FC matrix
corr_mat = np.ma.array(corr_mat, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
corr_mat = np.ma.ravel(corr_mat)

# remove masked elements from cross-correlation matrix
corr_mat = np.ma.compressed(corr_mat)

# Show the plots on the screen
plt.show()
