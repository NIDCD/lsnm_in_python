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
#   This file (compute_xcorr_diffs.py) was created on November 12, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on December 2 2016
#
# **************************************************************************/
#
# compute_xcorr_diffs.py
#
# Reads cross-correlation matrices from two separate input files and substracts
# one from the other. Displays a matrix of differences and a histogram. We also
# calculate kurtosis and skewness of the histogram.
#
# It saves the matrix of the correlation differences between Task-Based cross-correlation
# matrix and the Resting-State cross-correlation matrix
#
# Finally, we display a scatter plot of correlation coefficients of array1 vs array2

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm as CM

from scipy.stats import kurtosis

from scipy.stats import skew

# prepare labels
labels =  ['rLOF',     
    'rPORB',         
    'rFP'  ,          
    'rMOF' ,          
    'rPTRI',          
    'rPOPE',          
    'rRMF' ,          
    'rSF'  ,          
    'rCMF' ,          
    'rPREC',          
    'rPARC',          
    'rRAC' ,          
    'rCAC' ,          
    'rPC'  ,          
    'rISTC',          
    'rPSTC',          
    'rSMAR',          
    'rSP'  ,          
    'rIP'  ,          
    'rPCUN',          
    'rCUN' ,          
    'rPCAL',          
    'rLOCC',          
    'rLING',          
    'rFUS' ,          
    'rPARH',          
    'rENT' ,          
    'rTP'  ,          
    'rIT'  ,          
    'rMT'  ,          
    'rBSTS',          
    'rST'  ,          
    'rTT']            

# define the name and location of files where the cross-correlation matrices
# are stored
xcorr_file1 = 'output.RestingState/xcorr_matrix_33_regions.npy'
xcorr_file2 = 'output.TaskBased/xcorr_matrix_33_regions.npy'

xcorr_diff_file = 'xcorr_diffs_TB_minus_RS.npy'

# read the input files that contain cross-correlation matrices
xcorr1 = np.load(xcorr_file1)
xcorr2 = np.load(xcorr_file2)

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to subtracting.
xcorr1_Z = np.arctanh(xcorr1)
xcorr2_Z = np.arctanh(xcorr2)

# calculate correlation matrix difference:
xcorr_diff_Z = xcorr2_Z - xcorr1_Z

# now, convert back to from Z to R correlation coefficients, prior to plotting
xcorr_diff  = np.tanh(xcorr_diff_Z)

# save cross-correlation differences to an npy python file
np.save(xcorr_diff_file, xcorr_diff)


# initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)

# decrease font size
#plt.rcParams.update({'font.size': 8})

# plot correlation matrix as a heatmap
mask = np.tri(xcorr_diff.shape[0], k=0)
mask = np.transpose(mask)
xcorr_diff = np.ma.array(xcorr_diff, mask=mask)          # mask out the upper triangle
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w') 
cax = ax.imshow(xcorr_diff, vmin=-0.5, vmax=0.5, interpolation='nearest', cmap=cmap)
ax.grid(False)
plt.colorbar(cax)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

# initialize new figure for plotting histogram
fig = plt.figure()
ax = fig.add_subplot(111)

# flatten the numpy cross-correlation matrices
xcorr_diff = np.ma.ravel(xcorr_diff)
xcorr1 = np.ma.ravel(xcorr1)
xcorr2 = np.ma.ravel(xcorr2)

# remove NaN elements from cross-correlation matrices (they are along the diagonal)
xcorr_diff = xcorr_diff[~np.isnan(xcorr_diff)]
xcorr1 = xcorr1[~np.isnan(xcorr1)]
xcorr2 = xcorr2[~np.isnan(xcorr2)]

# remove masked elements from cross-correlation matrices
xcorr_diff = np.ma.compressed(xcorr_diff)
xcorr1 = np.ma.compressed(xcorr1)
xcorr2 = np.ma.compressed(xcorr2)

# plot a histogram to show the frequency of correlations in difference matrix
plt.hist(xcorr_diff, 16)
plt.xlabel('Correlation Coefficient Difference')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 200])

# calculate and print kurtosis
print 'Fishers kurtosis: ', kurtosis(xcorr_diff, fisher=True)
print 'Skewness: ', skew(xcorr_diff)

# initialize new figure for plotting scatter plot of xcorr1 vs xcorr2
fig = plt.figure()
ax = fig.add_subplot(111)

# plot a scatter plot to show how xcorr1 and xcorr2 correlate
plt.scatter(xcorr1, xcorr2)
plt.xlabel('Resting-State')
plt.ylabel('Task-Based')
plt.axis([-1,1.5,-1,1.5])

# Show the plots on the screen
plt.show()



