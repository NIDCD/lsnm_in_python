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
#   This file (avg_FC_TVB_ROIs_across_subjects.py) was created on November 30 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on November 30 2016
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/
#
# avg_FC_TVB_ROIs_across_subjects.py
#
# Reads the correlation coefficients (functional connectivity matrix) from
# several python (*.npy) data files, each
# corresponding to a single subject, and calculates:
# (1) the average functional connectivity across all subjects for two conditions:
#     Resting State and Task-based,
# (2) means, standard deviations and variances for each data point.
# Means, standard deviations, and variances are stored in a text output file.
# For the calculations, It uses
# previously calculated cross-correlation coefficients using simulated BOLD
# in Hagmann's connectome 33 ROIs (right hemisphere)
# It also performs a paired t-test for the comparison between the mean of each condition
# (task-based vs resting-state), and displays the t values. Note that in a paired t-test,
# each subject in the study is in both the treatment and the control group.

import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.tools as tls

import matplotlib as mpl

import pandas as pd

from scipy.stats import t

from scipy import stats

import math as m

from scipy.stats import kurtosis

from scipy.stats import skew

from matplotlib import cm as CM

# set matplot lib parameters to produce visually appealing plots
#mpl.style.use('ggplot')

# construct array of indices of modules contained in Hagmann's connectome
# (right hemisphere)
ROIs = np.arange(33)

# construct array of subjects to be considered
subjects = np.arange(10)

# define output file where means, standard deviations, and variances will be stored
RS_FC_avg_file = 'rs_fc_avg.npy'
TB_FC_avg_file = 'tb_fc_avg.npy'

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
FC_RS_subj1  = 'subject_11/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj2  = 'subject_12/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj3  = 'subject_13/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj4  = 'subject_14/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj5  = 'subject_15/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj6  = 'subject_16/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj7  = 'subject_17/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj8  = 'subject_18/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj9  = 'subject_19/output.RestingState/xcorr_matrix_33_regions.npy'
FC_RS_subj10 = 'subject_20/output.RestingState/xcorr_matrix_33_regions.npy'

FC_TB_subj1  = 'subject_11/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj2  = 'subject_12/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj3  = 'subject_13/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj4  = 'subject_14/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj5  = 'subject_15/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj6  = 'subject_16/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj7  = 'subject_17/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj8  = 'subject_18/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj9  = 'subject_19/output.TaskBased/xcorr_matrix_33_regions.npy'
FC_TB_subj10 = 'subject_20/output.TaskBased/xcorr_matrix_33_regions.npy'

# open files that contain correlation coefficients
fc_rs_subj1  = np.load(FC_RS_subj1)
fc_rs_subj2  = np.load(FC_RS_subj2)
fc_rs_subj3  = np.load(FC_RS_subj3)
fc_rs_subj4  = np.load(FC_RS_subj4)
fc_rs_subj5  = np.load(FC_RS_subj5)
fc_rs_subj6  = np.load(FC_RS_subj6)
fc_rs_subj7  = np.load(FC_RS_subj7)
fc_rs_subj8  = np.load(FC_RS_subj8)
fc_rs_subj9  = np.load(FC_RS_subj9)
fc_rs_subj10 = np.load(FC_RS_subj10)
fc_tb_subj1  = np.load(FC_TB_subj1)
fc_tb_subj2  = np.load(FC_TB_subj2)
fc_tb_subj3  = np.load(FC_TB_subj3)
fc_tb_subj4  = np.load(FC_TB_subj4)
fc_tb_subj5  = np.load(FC_TB_subj5)
fc_tb_subj6  = np.load(FC_TB_subj6)
fc_tb_subj7  = np.load(FC_TB_subj7)
fc_tb_subj8  = np.load(FC_TB_subj8)
fc_tb_subj9  = np.load(FC_TB_subj9)
fc_tb_subj10 = np.load(FC_TB_subj10)

# construct numpy arrays that contain correlation coefficient arrays for all subjects
fc_rs = np.array([fc_rs_subj1, fc_rs_subj2, fc_rs_subj3,
                       fc_rs_subj4, fc_rs_subj5, fc_rs_subj6,
                       fc_rs_subj7, fc_rs_subj8, fc_rs_subj9,
                       fc_rs_subj10 ]) 
fc_tb = np.array([fc_tb_subj1, fc_tb_subj2, fc_tb_subj3,
                       fc_tb_subj4, fc_tb_subj5, fc_tb_subj6,
                       fc_tb_subj7, fc_tb_subj8, fc_tb_subj9,
                       fc_tb_subj10 ]) 

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
fc_rs_z  = np.arctanh(fc_rs)
fc_tb_z  = np.arctanh(fc_tb)

# calculate the mean of correlation coefficients across all given subjects
fc_rs_z_mean = np.mean(fc_rs_z, axis=0)
fc_tb_z_mean = np.mean(fc_tb_z, axis=0)

# calculate the standard error of the mean of correlation coefficients across subjects
fc_rs_z_std = np.std(fc_rs_z, axis=0)
fc_tb_z_std = np.std(fc_tb_z, axis=0)

# calculate the variance of the correlation coefficients across subjects
fc_rs_z_var = np.var(fc_rs_z, axis=0)
fc_tb_z_var = np.var(fc_tb_z, axis=0)

# Calculate the statistical significance by using a two-tailed paired t-test:
# We are going to have one group of 10 subjects, doing DMS task(TB) and Resting State (RS)
# STEPS:
#     (1) Set up hypotheses:
# The NULL hypothesis is:
#          * The mean difference between paired observations (TB and RS) is zero
#            In other words, H_0 : mean(TB) = mean(RS)
# Our alternative hypothesis is:
#          * The mean difference between paired observations (TB and RS) is not zero
#            In other words, H_A : mean(TB) =! mean(RS)
#     (2) Set a significance level:
#         alpha = 1 - confidence interval = 1 - 95% = 1 - 0.95 = 0.05 
alpha = 0.05
#     (3) What is the critical value and the rejection region?
n = 10                         # sample size
df = n  - 1                    # degrees-of-freedom = n minus 1
                               # sample size is 10 because there are 10 subjects in each condition
rejection_region = 2.262       # as found on t-test table for t and dof given,
                               # null-hypothesis (H_0) will be rejected for those t above rejection_region
#     (4) compute the value of the test statistic                               
# calculate differences between the pairs of data:
d  = fc_tb_z - fc_rs_z
# calculate the mean of those differences
d_mean = np.mean(d, axis=0)
# calculate the standard deviation of those differences
d_std = np.std(d, axis=0)
# calculate square root of sample size
sqrt_n = m.sqrt(n)
# calculate standard error of the mean differences
d_sem = d_std/sqrt_n 
# calculate the t statistic:
t_star = d_mean / d_sem

# threshold and binarize the array of t statistics:
t_star_mask = np.where(t_star>rejection_region, 0, 1)

#initialize new figure for t-test values
fig = plt.figure('T-test values')
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(t_star.shape[0], k=0)
mask = np.transpose(mask)
t_star = np.ma.array(t_star, mask=mask)          # mask out upper triangle
t_star = np.ma.array(t_star, mask=t_star_mask)   # mask out elements rejected by t-test

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w')
cax = ax.imshow(t_star, interpolation='nearest', cmap=cmap)
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

# now, convert back to from Z to R correlation coefficients
fc_rs_mean  = np.tanh(fc_rs_z_mean)
fc_tb_mean  = np.tanh(fc_tb_z_mean)

fc_rs_std  = np.tanh(fc_rs_z_mean)
fc_tb_std  = np.tanh(fc_tb_z_mean)

fc_rs_mean  = np.tanh(fc_rs_z_mean)
fc_tb_mean  = np.tanh(fc_tb_z_mean)

# save the averages of both resting state and task based functional connectivity arrays
np.save(RS_FC_avg_file, fc_rs_mean)
np.save(TB_FC_avg_file, fc_tb_mean)

#initialize new figure for correlations of Resting State mean
fig = plt.figure('Mean Resting-State functional connectivity')
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(fc_rs_mean.shape[0], k=0)
mask = np.transpose(mask)
fc_rs_mean = np.ma.array(fc_rs_mean, mask=mask)    # mask out upper triangle

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w')
cax = ax.imshow(fc_rs_mean, vmin=-1, vmax=1, interpolation='nearest', cmap=cmap)
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

#initialize new figure for correlations of Task-based mean
fig = plt.figure('Mean Task-based Functional Connectivity')
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including diagonal
mask = np.tri(fc_tb_mean.shape[0], k=0)
mask = np.transpose(mask)
fc_tb_mean = np.ma.array(fc_tb_mean, mask=mask)    # mask out upper triangle

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w')
cax = ax.imshow(fc_tb_mean, vmin=-1, vmax=1, interpolation='nearest', cmap=cmap)
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


# initialize new figure for resting-state histogram
fig = plt.figure('Resting State')
ax = fig.add_subplot(111)

# flatten the numpy cross-correlation matrix
corr_mat = np.ma.ravel(fc_rs_mean)

# remove masked elements from cross-correlation matrix
corr_mat = np.ma.compressed(corr_mat)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 100])

# calculate and print kurtosis
print '\nResting-State Fishers kurtosis: ', kurtosis(corr_mat, fisher=True)
print 'Resting-State Skewness: ', skew(corr_mat)

# initialize new figure for Task-Based plotting histogram
fig = plt.figure('Task Based')
ax = fig.add_subplot(111)

# flatten the numpy cross-correlation matrix
corr_mat = np.ma.ravel(fc_tb_mean)

# remove masked elements from cross-correlation matrix
corr_mat = np.ma.compressed(corr_mat)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 100])

# calculate and print kurtosis
print 'Task-based Fishers kurtosis: ', kurtosis(corr_mat, fisher=True)
print 'Task-based Skewness: ', skew(corr_mat)


# Show the plots on the screen
plt.show()
