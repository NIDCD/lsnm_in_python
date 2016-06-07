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
#   This file (compute_PSC.py) was created on December 2, 2015.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on February 4, 2016  
# **************************************************************************/

# compute_PSC.py
#
# Calculate and plot the Percent Signal Change (PSC) of fMRI BOLD timeseries across
# subjects for 2 conditions: DMS and passive viewing.
#
# The inputs are BOLD timeseries for each subject, one timeseries per brain module.
# We take each timeseries and convert
# each to PSC by dividing every time point by the baseline (the mean of the whole
# time course for a given subject and brain area), then multiplying by 100
# (Source: BrainVoyager v20.0 User's Guide). That way,
# we obtain a
# whole-experiment timecourse in percent signal change, per brain area, per subject.
# Then, we average together all time points for each condition, DMS and control,
# separately, across subjects, per brain area. That gives us one PSC number per
# brain area per condition, for all subjects.
# 
# We also do a paired t-test to check for the statistical significance of the difference
# between DMS and control conditions, per brain area.
#

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

import math as math

from scipy.stats import poisson

from scipy.stats import t

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110
num_of_trials = 36             # number of trials per subject

num_of_fmri_blocks = 12             # how many blocks of trials in the experiment
num_of_syn_blocks = 12              # we have more synaptic blocks than fmri blocks
                                    # because we get rid of blocks in BOLD timeseries

num_of_subjects = 10

scans_removed = 0             # number of scans removed from BOLD computation
synaptic_steps_removed = 0    # number of synaptic steps removed from synaptic
                              # computation

num_of_modules = 8            # V1, V4, IT, FS, D1, D2, FR, cIT (contralateral IT)

# define total number of trials across subjects
total_trials = num_of_trials * num_of_subjects

# define total number of blocks across subjects
total_fmri_blocks = num_of_fmri_blocks * num_of_subjects
#total_syn_blocks  = num_of_syn_blocks * num_of_subjects

synaptic_timesteps = experiment_length - synaptic_steps_removed

# define the names of the input files where the BOLD timeseries are contained
BOLD_ts_subj=np.array(['subject_11/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_12/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_13/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_14/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_15/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_16/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_17/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_18/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_19/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
                       'subject_20/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy'])

# set matplot lib parameters to produce visually appealing plots
#mpl.style.use('ggplot')

# define output file where means, standard deviations, and variances will be stored
PSC_stats_FILE = 'psc_stats_TVB_ROI.txt'

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 5 ms.
Ti = 0.005 * 10

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# Time for one complete trial in milliseconds
Ttrial = 5.5

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

num_of_scans = T / Tr - scans_removed
                      
# Construct a numpy array of time points
time_in_seconds = np.arange(0, T, Tr)

scanning_timescale = np.arange(0, synaptic_timesteps, synaptic_timesteps / (T/Tr))

# open files containing BOLD time-series
BOLD_ts = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
for idx in range(0, num_of_subjects):
    BOLD_ts[idx] = np.load(BOLD_ts_subj[idx])    

# Perform time-course normalization by converting each timeseries to percent signal change
# for each subject and for each module relative to the mean of each time course.
for s in range(0, num_of_subjects):
    for m in range(0, num_of_modules):
        timecourse_mean = np.mean(BOLD_ts[s,m])
        BOLD_ts[s,m] = BOLD_ts[s,m] / timecourse_mean * 100. - 100.

mean_PSC_dms = np.zeros((num_of_subjects, num_of_modules))
mean_PSC_ctl = np.zeros((num_of_subjects, num_of_modules))
scans_in_each_block = int(round(num_of_scans / (num_of_fmri_blocks / 2.), 0))
print 'Scans in each block: ', scans_in_each_block
# now, perform a mean of PSCs of scan 4, 5, 6 of each condition
for s in range(0, num_of_subjects):
    for m in range(0, num_of_modules):
        for i in range(0, num_of_scans, scans_in_each_block):
            mean_PSC_dms[s,m] = (BOLD_ts[s,m,i+4] + BOLD_ts[s,m,i+5] + BOLD_ts[s,m,i+6]) / 3.
            mean_PSC_ctl[s,m] = (BOLD_ts[s,m,i+11]+ BOLD_ts[s,m,i+12]+ BOLD_ts[s,m,i+13]) / 3.

# now, calculate the average PSC across subjects for each brain region and for each condition:
BOLD_v1_dms_avg = np.mean(mean_PSC_dms[:,0]) 
BOLD_v4_dms_avg = np.mean(mean_PSC_dms[:,1])
BOLD_it_dms_avg = np.mean(mean_PSC_dms[:,2])
BOLD_fs_dms_avg = np.mean(mean_PSC_dms[:,3])
BOLD_d1_dms_avg = np.mean(mean_PSC_dms[:,4])
BOLD_d2_dms_avg = np.mean(mean_PSC_dms[:,5])
BOLD_fr_dms_avg = np.mean(mean_PSC_dms[:,6])
BOLD_v1_ctl_avg = np.mean(mean_PSC_ctl[:,0])
BOLD_v4_ctl_avg = np.mean(mean_PSC_ctl[:,1])
BOLD_it_ctl_avg = np.mean(mean_PSC_ctl[:,2])
BOLD_fs_ctl_avg = np.mean(mean_PSC_ctl[:,3])
BOLD_d1_ctl_avg = np.mean(mean_PSC_ctl[:,4])
BOLD_d2_ctl_avg = np.mean(mean_PSC_ctl[:,5])
BOLD_fr_ctl_avg = np.mean(mean_PSC_ctl[:,6])

# calculate the variance as well, across subjects, for each brain regions and for each condition:
BOLD_v1_dms_var = np.var(mean_PSC_dms[:,0])
BOLD_v4_dms_var = np.var(mean_PSC_dms[:,1])
BOLD_it_dms_var = np.var(mean_PSC_dms[:,2])
BOLD_fs_dms_var = np.var(mean_PSC_dms[:,3])
BOLD_d1_dms_var = np.var(mean_PSC_dms[:,4])
BOLD_d2_dms_var = np.var(mean_PSC_dms[:,5])
BOLD_fr_dms_var = np.var(mean_PSC_dms[:,6])
BOLD_v1_ctl_var = np.var(mean_PSC_ctl[:,0])
BOLD_v4_ctl_var = np.var(mean_PSC_ctl[:,1])
BOLD_it_ctl_var = np.var(mean_PSC_ctl[:,2])
BOLD_fs_ctl_var = np.var(mean_PSC_ctl[:,3])
BOLD_d1_ctl_var = np.var(mean_PSC_ctl[:,4])
BOLD_d2_ctl_var = np.var(mean_PSC_ctl[:,5])
BOLD_fr_ctl_var = np.var(mean_PSC_ctl[:,6])
        
# calculate the standard deviation of the PSC across subjects, for each brain region and for each
# condition
BOLD_v1_dms_std = np.std(mean_PSC_dms[:,0])
BOLD_v4_dms_std = np.std(mean_PSC_dms[:,1])
BOLD_it_dms_std = np.std(mean_PSC_dms[:,2])
BOLD_fs_dms_std = np.std(mean_PSC_dms[:,3])
BOLD_d1_dms_std = np.std(mean_PSC_dms[:,4])
BOLD_d2_dms_std = np.std(mean_PSC_dms[:,5])
BOLD_fr_dms_std = np.std(mean_PSC_dms[:,6])
BOLD_v1_ctl_std = np.std(mean_PSC_ctl[:,0])
BOLD_v4_ctl_std = np.std(mean_PSC_ctl[:,1])
BOLD_it_ctl_std = np.std(mean_PSC_ctl[:,2])
BOLD_fs_ctl_std = np.std(mean_PSC_ctl[:,3])
BOLD_d1_ctl_std = np.std(mean_PSC_ctl[:,4])
BOLD_d2_ctl_std = np.std(mean_PSC_ctl[:,5])
BOLD_fr_ctl_std = np.std(mean_PSC_ctl[:,6])

# Display all PSCs for both DMS and CTL, which represent the average of
# the Percent Signal Change per each brain area per each module
print 'BOLD V1 DMS mean, std, var: [', BOLD_v1_dms_avg, BOLD_v1_dms_std, BOLD_v1_dms_var, ']'
print 'BOLD V4 DMS mean, std, var: [', BOLD_v4_dms_avg, BOLD_v4_dms_std, BOLD_v4_dms_var, ']'
print 'BOLD IT DMS mean, std, var: [', BOLD_it_dms_avg, BOLD_it_dms_std, BOLD_it_dms_var, ']'
print 'BOLD FS DMS mean, std, var: [', BOLD_fs_dms_avg, BOLD_fs_dms_std, BOLD_fs_dms_var, ']'
print 'BOLD D1 DMS mean, std, var: [', BOLD_d1_dms_avg, BOLD_d1_dms_std, BOLD_d1_dms_var, ']'
print 'BOLD D2 DMS mean, std, var: [', BOLD_d2_dms_avg, BOLD_d2_dms_std, BOLD_d2_dms_var, ']'
print 'BOLD FR DMS mean, std, var: [', BOLD_fr_dms_avg, BOLD_fr_dms_std, BOLD_fr_dms_var, ']'
print 'BOLD V1 CTL mean, std, var: [', BOLD_v1_ctl_avg, BOLD_v1_ctl_std, BOLD_v1_ctl_var, ']'
print 'BOLD V4 CTL mean, std, var: [', BOLD_v4_ctl_avg, BOLD_v4_ctl_std, BOLD_v4_ctl_var, ']'
print 'BOLD IT CTL mean, std, var: [', BOLD_it_ctl_avg, BOLD_it_ctl_std, BOLD_it_ctl_var, ']'
print 'BOLD FS CTL mean, std, var: [', BOLD_fs_ctl_avg, BOLD_fs_ctl_std, BOLD_fs_ctl_var, ']'
print 'BOLD D1 CTL mean, std, var: [', BOLD_d1_ctl_avg, BOLD_d1_ctl_std, BOLD_d1_ctl_var, ']'
print 'BOLD D2 CTL mean, std, var: [', BOLD_d2_ctl_avg, BOLD_d2_ctl_std, BOLD_d2_ctl_var, ']'
print 'BOLD FR CTL mean, std, var: [', BOLD_fr_ctl_avg, BOLD_fr_ctl_std, BOLD_fr_ctl_var, ']'


# Calculate the statistical significance by using a one-tailed paired t-test:
# We are going to have one group of 10 subjects, doing both DMS and control task
# STEPS:
#     (1) Set up hypotheses:
# The NULL hypothesis is:
#          * The mean difference between paired observations (DMS and CTL) is zero
# Our alternative hypothesis is:
#          * The mean difference between paired observations (DMS and CTL) is not zero
#     (2) Set a significance level:
alpha = 0.05
#     (3) What is the critical value and the rejection region?
n = 10 - 1                   # sample size minus 1
rejection_region = 1.833       # as found on t-test table for t and dof given,
                               # values of t above rejection_region will be rejected
#     (4) compute the value of the test statistic                               
# calculate differences between the pairs of data:
d  = mean_PSC_dms - mean_PSC_ctl
# calculate the mean of those differences
d_mean = np.mean(d, axis=0)
# calculate the standard deviation of those differences
d_std = np.std(d, axis=0)
# calculate square root of sample size minus 1
sqrt_n = math.sqrt(n)
# calculate standard error of the mean differences
d_sem = d_std/sqrt_n 
# calculate the t statistic:
t_star = d_mean / d_sem

print 'Mean differences: ', d_mean

print 't-values for PSC comparisons (V1, V4, IT, FS, D1, D2, FR, cIT): ', t_star

print 'Dimensions of mean differences array', d_mean.shape
print 'Dimensions of std of differences array', d_std.shape

# increase font size prior to plotting
plt.rcParams.update({'font.size': 15})

# define number of groups to plot
N = 1

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.2                     # width of the bars

fig, ax = plt.subplots()

#ax.set_ylim([0,3.5])

# now, group the values to be plotted by brain module and by task condition

rects_v1_dms = ax.bar(index, d_mean[0], width, color='yellow',
                      label='V1', yerr=d_sem[0] )

rects_v4_dms = ax.bar(index + width, d_mean[1], width, color='green',
                      label='V4', yerr=d_sem[1] )

rects_it_dms = ax.bar(index + width*2, d_mean[2], width, color='blue',
                      label='IT', yerr=d_sem[2] )

rects_fs_dms = ax.bar(index + width*3, d_mean[3], width, color='orange',
                      label='FS', yerr=d_sem[3] )

rects_d1_dms = ax.bar(index + width*4, d_mean[4], width, color='red',
                      label='D1', yerr=d_sem[4] )

rects_d2_dms = ax.bar(index + width*5, d_mean[5], width, color='pink',
                      label='D2', yerr=d_sem[5] )

rects_fr_dms = ax.bar(index + width*6, d_mean[6], width, color='purple',
                      label='FR', yerr=d_sem[6] )

#ax.set_title('PSC ACROSS SUBJECTS IN ALL BRAIN REGIONS')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.set_ylabel('Signal change differences')

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

#set up figure to plot BOLD signal (normalized to PSC)
fig=plt.figure(2)

# plot V1 BOLD time-series in yellow
ax = plt.subplot(7,1,1)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[0], linewidth=3.0, color='yellow')
# plot V4 BOLD time-series in yellow
ax = plt.subplot(7,1,2)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[1], linewidth=3.0, color='lime')
# plot IT BOLD time-series in yellow
ax = plt.subplot(7,1,3)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[2], linewidth=3.0, color='blue')
# plot FS BOLD time-series in yellow
ax = plt.subplot(7,1,4)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[3], linewidth=3.0, color='orange')
# plot D1 BOLD time-series in yellow
ax = plt.subplot(7,1,5)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[4], linewidth=3.0, color='red')
# plot D2 BOLD time-series in yellow
ax = plt.subplot(7,1,6)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[5], linewidth=3.0, color='pink')
# plot FR BOLD time-series in yellow
ax = plt.subplot(7,1,7)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(BOLD_ts[6], linewidth=3.0, color='darkorchid')


# Show the plots on the screen
plt.show()

