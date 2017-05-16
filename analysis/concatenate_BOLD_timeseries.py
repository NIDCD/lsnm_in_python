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
#   This file (concatenate_BOLD_timeseries.py) was created on May 9 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on May 9 2017
#
# **************************************************************************/

# concatenate_BOLD_timeseries.py
#
# concatenate Passive Viewing (PV) and DMS Task BOLD fMRI timeseries for each one
# of 998 ROIs, then compute, display and save the fucntional connectivity matrices
# using the new PV + DMS timeseries

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm as CM

# define the location/name of the two input files(containing BOLD timeseries) to
# be concatenated
BOLD_file_1 = 'output.PassiveViewing/bold_balloon_998_regions.npy'
BOLD_file_2 = 'output.DMSTask/bold_balloon_998_regions.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
xcorr_file = 'PV_DMS_xcorr_matrix_998_regions.npy'

# read the input file that contains the synaptic activities of all ROIs
BOLD_1 = np.load(BOLD_file_1)
BOLD_2 = np.load(BOLD_file_2)

# claculate total number of ROIs
ROIs = BOLD_1.shape[0]

print 'Dimensions of BOLD file: ', BOLD_1.shape
print 'LENGTH OF each BOLD TIME-SERIES: ', BOLD_1[0].size
print 'Number of ROIs: ', ROIs

BOLD_c = np.concatenate((BOLD_1, BOLD_2), axis=1)

print 'Size of BOLD time-series after concatenating: ', BOLD_c.shape

plt.figure()

plt.suptitle('CONCATENATED PV + DMS BOLD SIGNAL')

# plot BOLD time-series for all ROIs
for idx, roi in enumerate(BOLD_c):
    plt.plot(BOLD_c[idx], linewidth=1.0, color='black')

# calculate correlation matrix
corr_mat = np.corrcoef(BOLD_c)
print 'Number of ROIs: ', BOLD_c.shape
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
