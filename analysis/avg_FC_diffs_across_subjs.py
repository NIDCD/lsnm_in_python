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
#   This file (avg_FC_diffs_across_subjs.py) was created on December 3, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on December 3 2016
#
# **************************************************************************/
#
# avg_FC_diffs_across_subjs.py
#
# Reads functional connectivity differences from input files corresponding to
# difference subjects and it calculates an average, after which it displays
# a average in matrix form as well as a histogram. We also
# calculate kurtosis and skewness of the histogram.


import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kurtosis

from scipy.stats import skew

from matplotlib import cm as CM

# declare ROI labels
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


# define the names of the input files where the correlation coefficients were stored
FC_diff_subj1  = 'subject_11/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj2  = 'subject_12/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj3  = 'subject_13/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj4  = 'subject_14/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj5  = 'subject_15/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj6  = 'subject_16/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj7  = 'subject_17/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj8  = 'subject_18/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj9  = 'subject_19/xcorr_diffs_TB_minus_RS.npy'
FC_diff_subj10 = 'subject_20/xcorr_diffs_TB_minus_RS.npy'

# define the names of the average fucntional connectivies are stored (for both
# task-based and resting state)
xcorr_rs_avg_file = 'rs_fc_avg.npy'
xcorr_tb_avg_file = 'tb_fc_avg.npy'

# open files that contain correlation coefficients
fc_diff_subj1  = np.load(FC_diff_subj1)
fc_diff_subj2  = np.load(FC_diff_subj2)
fc_diff_subj3  = np.load(FC_diff_subj3)
fc_diff_subj4  = np.load(FC_diff_subj4)
fc_diff_subj5  = np.load(FC_diff_subj5)
fc_diff_subj6  = np.load(FC_diff_subj6)
fc_diff_subj7  = np.load(FC_diff_subj7)
fc_diff_subj8  = np.load(FC_diff_subj8)
fc_diff_subj9  = np.load(FC_diff_subj9)
fc_diff_subj10 = np.load(FC_diff_subj10)

# open files that contain functional connectivity averages
xcorr_rs_avg = np.load(xcorr_rs_avg_file)
xcorr_tb_avg = np.load(xcorr_tb_avg_file)

# construct numpy array containing functional connectivity arrays
fc_diff = np.array([fc_diff_subj1, fc_diff_subj2, fc_diff_subj3,
                    fc_diff_subj4, fc_diff_subj5, fc_diff_subj6,
                    fc_diff_subj7, fc_diff_subj8, fc_diff_subj9,
                    fc_diff_subj10 ]) 

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
fc_diff_z  = np.arctanh(fc_diff)
fc_diff_z  = np.arctanh(fc_diff)

# calculate the mean of correlation coefficients across all given subjects
fc_diff_z_mean = np.mean(fc_diff_z, axis=0)
fc_diff_z_mean = np.mean(fc_diff_z, axis=0)

# now, convert back to from Z to R correlation coefficients, prior to plotting
fc_diff_mean  = np.tanh(fc_diff_z_mean)
fc_diff_mean  = np.tanh(fc_diff_z_mean)

#initialize new figure for correlations of Resting State mean
fig = plt.figure('Across-subject average of FC differences (TB-RS)')
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(fc_diff_mean.shape[0], k=0)
mask = np.transpose(mask)
fc_diff_mean = np.ma.array(fc_diff_mean, mask=mask)    # mask out upper triangle

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w')
cax = ax.imshow(fc_diff_mean, vmin=-1, vmax=1.0, interpolation='nearest', cmap=cmap)
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

# initialize new figure for histogram
fig = plt.figure('Average of FC differences (TB-RS)')
ax = fig.add_subplot(111)

# flatten the numpy cross-correlation matrix
corr_mat = np.ma.ravel(fc_diff_mean)

# remove masked elements from cross-correlation matrix
corr_mat = np.ma.compressed(corr_mat)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 130])

# initialize new figure scatter plot of xcorr_rs average vs xcorr_tb average
fig = plt.figure()
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(xcorr_rs_avg.shape[0], k=0)
mask = np.transpose(mask)
xcorr_rs_avg = np.ma.array(xcorr_rs_avg, mask=mask)    # mask out upper triangle
xcorr_tb_avg = np.ma.array(xcorr_tb_avg, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation arrays
corr_mat_RS = np.ma.ravel(xcorr_rs_avg)
corr_mat_TB = np.ma.ravel(xcorr_tb_avg)

# remove masked elements from cross-correlation arrays
corr_mat_RS = np.ma.compressed(xcorr_rs_avg)
corr_mat_TB = np.ma.compressed(xcorr_tb_avg)


# plot a scatter plot to show how averages of xcorr1 and xcorr2 correlate
plt.scatter(corr_mat_RS, corr_mat_TB)
plt.xlabel('Resting-State')
plt.ylabel('Task-Based')
plt.axis([-1,1,-1,1])

# calculate and print kurtosis
print '\nResting-State Fishers kurtosis: ', kurtosis(corr_mat, fisher=True)
print 'Resting-State Skewness: ', skew(corr_mat)


# Show the plots on the screen
plt.show()
