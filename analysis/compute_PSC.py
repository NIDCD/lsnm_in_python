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
# We also do a t-test to check for the statistical significance of the difference
# between DMS and control conditions, per brain area.
#

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

import math as m

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
BOLD_ts_subj=np.array(['subject_11/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_12/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_13/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_14/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_15/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_16/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_17/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_18/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_19/output.36trials.with_feedback/lsnm_bold_balloon.npy',
                       'subject_20/output.36trials.with_feedback/lsnm_bold_balloon.npy'])

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# define output file where means, standard deviations, and variances will be stored
fc_stats_FILE = 'fc_stats.txt'

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
# now, perform a mean of PSCs of scan 3, 4, 5 of each condition
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

# Concatenate the time-series of each module together, across all subjects:
# BOLD_ts = np.reshape(BOLD_ts, (num_of_modules, num_of_subjects*num_of_scans))

# now, split all time-series in blocks
#BOLD_ts_v1=np.array_split(BOLD_ts[0], total_fmri_blocks)
#BOLD_ts_v4=np.array_split(BOLD_ts[1], total_fmri_blocks)
#BOLD_ts_it=np.array_split(BOLD_ts[2], total_fmri_blocks)
#BOLD_ts_fs=np.array_split(BOLD_ts[3], total_fmri_blocks)
#BOLD_ts_d1=np.array_split(BOLD_ts[4], total_fmri_blocks)
#BOLD_ts_d2=np.array_split(BOLD_ts[5], total_fmri_blocks)
#BOLD_ts_fr=np.array_split(BOLD_ts[6], total_fmri_blocks)

# define an array with location of control blocks, and another array
# with location of task (DMS) blocks, relative to
# an array that contains all blocks (task-related blocks included)
#fmri_control_block_ids = np.arange(1, total_fmri_blocks, 2)
#fmri_dms_block_ids =     np.arange(0, total_fmri_blocks, 2)

# now, create an array of BOLD time-series containing DMS trials only: 
#BOLD_v1_dms_blocks = np.delete(np.asarray(BOLD_ts_v1), fmri_control_block_ids, axis=0)
#BOLD_v4_dms_blocks = np.delete(np.asarray(BOLD_ts_v4), fmri_control_block_ids, axis=0)
#BOLD_it_dms_blocks = np.delete(np.asarray(BOLD_ts_it), fmri_control_block_ids, axis=0)
#BOLD_fs_dms_blocks = np.delete(np.asarray(BOLD_ts_fs), fmri_control_block_ids, axis=0)
#BOLD_d1_dms_blocks = np.delete(np.asarray(BOLD_ts_d1), fmri_control_block_ids, axis=0)
#BOLD_d2_dms_blocks = np.delete(np.asarray(BOLD_ts_d2), fmri_control_block_ids, axis=0)
#BOLD_fr_dms_blocks = np.delete(np.asarray(BOLD_ts_fr), fmri_control_block_ids, axis=0)

# ... and concatenate those DMS BOLD timeseries together
#BOLD_v1_dms_ts = np.concatenate(BOLD_v1_dms_blocks)
#BOLD_v4_dms_ts = np.concatenate(BOLD_v4_dms_blocks)
#BOLD_it_dms_ts = np.concatenate(BOLD_it_dms_blocks)
#BOLD_fs_dms_ts = np.concatenate(BOLD_fs_dms_blocks)
#BOLD_d1_dms_ts = np.concatenate(BOLD_d1_dms_blocks)
#BOLD_d2_dms_ts = np.concatenate(BOLD_d2_dms_blocks)
#BOLD_fr_dms_ts = np.concatenate(BOLD_fr_dms_blocks)

# but also, get rid of the DMS blocks, to create arrays with only control trials
#BOLD_v1_control_blocks = np.delete(np.asarray(BOLD_ts_v1), fmri_dms_block_ids, axis=0)
#BOLD_v4_control_blocks = np.delete(np.asarray(BOLD_ts_v4), fmri_dms_block_ids, axis=0)
#BOLD_it_control_blocks = np.delete(np.asarray(BOLD_ts_it), fmri_dms_block_ids, axis=0)
#BOLD_fs_control_blocks = np.delete(np.asarray(BOLD_ts_fs), fmri_dms_block_ids, axis=0)
#BOLD_d1_control_blocks = np.delete(np.asarray(BOLD_ts_d1), fmri_dms_block_ids, axis=0)
#BOLD_d2_control_blocks = np.delete(np.asarray(BOLD_ts_d2), fmri_dms_block_ids, axis=0)
#BOLD_fr_control_blocks = np.delete(np.asarray(BOLD_ts_fr), fmri_dms_block_ids, axis=0)

# ... and concatenate the control blocks together
#BOLD_v1_ctl_ts = np.concatenate(BOLD_v1_control_blocks)
#BOLD_v4_ctl_ts = np.concatenate(BOLD_v4_control_blocks)
#BOLD_it_ctl_ts = np.concatenate(BOLD_it_control_blocks)
#BOLD_fs_ctl_ts = np.concatenate(BOLD_fs_control_blocks)
#BOLD_d1_ctl_ts = np.concatenate(BOLD_d1_control_blocks)
#BOLD_d2_ctl_ts = np.concatenate(BOLD_d2_control_blocks)
#BOLD_fr_ctl_ts = np.concatenate(BOLD_fr_control_blocks)

# Average all timepoints together for the DMS condition:
#BOLD_v1_dms_avg = np.mean(BOLD_v1_dms_ts)
#BOLD_v4_dms_avg = np.mean(BOLD_v4_dms_ts)
#BOLD_it_dms_avg = np.mean(BOLD_it_dms_ts)
#BOLD_fs_dms_avg = np.mean(BOLD_fs_dms_ts)
#BOLD_d1_dms_avg = np.mean(BOLD_d1_dms_ts)
#BOLD_d2_dms_avg = np.mean(BOLD_d2_dms_ts)
#BOLD_fr_dms_avg = np.mean(BOLD_fr_dms_ts)

# Average all timepoints together for the control condition:
#BOLD_v1_ctl_avg = np.mean(BOLD_v1_ctl_ts)
#BOLD_v4_ctl_avg = np.mean(BOLD_v4_ctl_ts)
#BOLD_it_ctl_avg = np.mean(BOLD_it_ctl_ts)
#BOLD_fs_ctl_avg = np.mean(BOLD_fs_ctl_ts)
#BOLD_d1_ctl_avg = np.mean(BOLD_d1_ctl_ts)
#BOLD_d2_ctl_avg = np.mean(BOLD_d2_ctl_ts)
#BOLD_fr_ctl_avg = np.mean(BOLD_fr_ctl_ts)

# Calculate variance for the DMS condition:
#BOLD_v1_dms_var = np.var(BOLD_v1_dms_ts)
#BOLD_v4_dms_var = np.var(BOLD_v4_dms_ts)
#BOLD_it_dms_var = np.var(BOLD_it_dms_ts)
#BOLD_fs_dms_var = np.var(BOLD_fs_dms_ts)
#BOLD_d1_dms_var = np.var(BOLD_d1_dms_ts)
#BOLD_d2_dms_var = np.var(BOLD_d2_dms_ts)
#BOLD_fr_dms_var = np.var(BOLD_fr_dms_ts)

 # Calculate variance for the control condition:
#BOLD_v1_ctl_var = np.var(BOLD_v1_ctl_ts)
#BOLD_v4_ctl_var = np.var(BOLD_v4_ctl_ts)
#BOLD_it_ctl_var = np.var(BOLD_it_ctl_ts)
#BOLD_fs_ctl_var = np.var(BOLD_fs_ctl_ts)
#BOLD_d1_ctl_var = np.var(BOLD_d1_ctl_ts)
#BOLD_d2_ctl_var = np.var(BOLD_d2_ctl_ts)
#BOLD_fr_ctl_var = np.var(BOLD_fr_ctl_ts)

# Display all PSCs for both DMS and CTL, which represent the average of
# the Percent Signal Change per each brain area per each module
print 'BOLD V1 DMS Avg: ', BOLD_v1_dms_avg
print 'BOLD V4 DMS Avg: ', BOLD_v4_dms_avg
print 'BOLD IT DMS Avg: ', BOLD_it_dms_avg
print 'BOLD FS DMS Avg: ', BOLD_fs_dms_avg
print 'BOLD D1 DMS Avg: ', BOLD_d1_dms_avg
print 'BOLD D2 DMS Avg: ', BOLD_d2_dms_avg
print 'BOLD FR DMS Avg: ', BOLD_fr_dms_avg
print 'BOLD V1 CTL Avg: ', BOLD_v1_ctl_avg
print 'BOLD V4 CTL Avg: ', BOLD_v4_ctl_avg
print 'BOLD IT CTL Avg: ', BOLD_it_ctl_avg
print 'BOLD FS CTL Avg: ', BOLD_fs_ctl_avg
print 'BOLD D1 CTL Avg: ', BOLD_d1_ctl_avg
print 'BOLD D2 CTL Avg: ', BOLD_d2_ctl_avg
print 'BOLD FR CTL Avg: ', BOLD_fr_ctl_avg

# Calculate statistical significance by using a two-tailed t-test:
# We are going to have two groups: DMS group and control (CTL) group (each sample size is 10 subjects)
# Our research hypothesis is:
#          * The BOLD signal in the DMS group IS larger than the BOLD signal in the CTL group.
# The NULL hypothesis is:
#          * The BOLD signal in the DMS group IS NOT larger than the BOLD signal in the CTL group.
# The value of alpha (p-threshold) will be 0.05
sample_size = num_of_subjects
print 'sample size = ', num_of_subjects
number_of_groups = 2
# STEPS:
#     (1) subtract the mean of control group from the mean of DMS group:
BOLD_v1_diff = BOLD_v1_dms_avg - BOLD_v1_ctl_avg
BOLD_v4_diff = BOLD_v4_dms_avg - BOLD_v4_ctl_avg
BOLD_it_diff = BOLD_it_dms_avg - BOLD_it_ctl_avg
BOLD_fs_diff = BOLD_fs_dms_avg - BOLD_fs_ctl_avg
BOLD_d1_diff = BOLD_d1_dms_avg - BOLD_d1_ctl_avg
BOLD_d2_diff = BOLD_d2_dms_avg - BOLD_d2_ctl_avg
BOLD_fr_diff = BOLD_fr_dms_avg - BOLD_fr_ctl_avg
#     (2) Calculate, for both control and DMS, the variance divided by sample size minus 1:
BOLD_v1_ctl_a= BOLD_v1_ctl_var / (sample_size-1)
BOLD_v4_ctl_a= BOLD_v4_ctl_var / (sample_size-1)
BOLD_it_ctl_a= BOLD_it_ctl_var / (sample_size-1)
BOLD_fs_ctl_a= BOLD_fs_ctl_var / (sample_size-1)
BOLD_d1_ctl_a= BOLD_d1_ctl_var / (sample_size-1)
BOLD_d2_ctl_a= BOLD_d2_ctl_var / (sample_size-1)
BOLD_fr_ctl_a= BOLD_fr_ctl_var / (sample_size-1)
BOLD_v1_dms_a= BOLD_v1_dms_var / (sample_size-1)
BOLD_v4_dms_a= BOLD_v4_dms_var / (sample_size-1)
BOLD_it_dms_a= BOLD_it_dms_var / (sample_size-1)
BOLD_fs_dms_a= BOLD_fs_dms_var / (sample_size-1)
BOLD_d1_dms_a= BOLD_d1_dms_var / (sample_size-1)
BOLD_d2_dms_a= BOLD_d2_dms_var / (sample_size-1)
BOLD_fr_dms_a= BOLD_fr_dms_var / (sample_size-1)
#     (3) Add results obtained for CTL and DMS in step (2) together:
BOLD_v1_a = BOLD_v1_ctl_a + BOLD_v1_dms_a
BOLD_v4_a = BOLD_v4_ctl_a + BOLD_v4_dms_a
BOLD_it_a = BOLD_it_ctl_a + BOLD_it_dms_a
BOLD_fs_a = BOLD_fs_ctl_a + BOLD_fs_dms_a
BOLD_d1_a = BOLD_d1_ctl_a + BOLD_d1_dms_a
BOLD_d2_a = BOLD_d2_ctl_a + BOLD_d2_dms_a
BOLD_fr_a = BOLD_fr_ctl_a + BOLD_fr_dms_a
#     (4) Take the square root the results in step (3):
sqrt_BOLD_v1_a= np.sqrt(BOLD_v1_a)
sqrt_BOLD_v4_a= np.sqrt(BOLD_v4_a)
sqrt_BOLD_it_a= np.sqrt(BOLD_it_a)
sqrt_BOLD_fs_a= np.sqrt(BOLD_fs_a)
sqrt_BOLD_d1_a= np.sqrt(BOLD_d1_a)
sqrt_BOLD_d2_a= np.sqrt(BOLD_d2_a)
sqrt_BOLD_fr_a= np.sqrt(BOLD_fr_a)
#     (5) Divide the results of step (1) by the results of step (4) to obtain 't':
BOLD_v1_t= BOLD_v1_diff / sqrt_BOLD_v1_a
BOLD_v4_t= BOLD_v4_diff / sqrt_BOLD_v4_a
BOLD_it_t= BOLD_it_diff / sqrt_BOLD_it_a
BOLD_fs_t= BOLD_fs_diff / sqrt_BOLD_fs_a
BOLD_d1_t= BOLD_d1_diff / sqrt_BOLD_d1_a
BOLD_d2_t= BOLD_d2_diff / sqrt_BOLD_d2_a
BOLD_fr_t= BOLD_fr_diff / sqrt_BOLD_fr_a
#     (6) Calculate the degrees of freedom (add up number of observations for each group
#         minus number of groups):
dof = sample_size + sample_size - number_of_groups
print 'Degrees of freedom: ', dof
#     (7) find the p-values for the above 't' and 'degrees of freedom':
BOLD_v1_p_values = t.sf(BOLD_v1_t, dof)
BOLD_v4_p_values = t.sf(BOLD_v4_t, dof)
BOLD_it_p_values = t.sf(BOLD_it_t, dof)
BOLD_fs_p_values = t.sf(BOLD_fs_t, dof)
BOLD_d1_p_values = t.sf(BOLD_d1_t, dof)
BOLD_d2_p_values = t.sf(BOLD_d2_t, dof)
BOLD_fr_p_values = t.sf(BOLD_fr_t, dof)

print 't-value for v1 BOLD signal difference: ', BOLD_v1_t
print 't-value for v4 BOLD signal difference: ', BOLD_v4_t
print 't-value for it BOLD signal difference: ', BOLD_it_t
print 't-value for fs BOLD signal difference: ', BOLD_fs_t
print 't-value for d1 BOLD signal difference: ', BOLD_d1_t
print 't-value for d2 BOLD signal difference: ', BOLD_d2_t
print 't-value for fr BOLD signal difference: ', BOLD_fr_t


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

rects_v1_dms = ax.bar(index, BOLD_v1_dms_avg, width, color='yellow',
                      label='V1 during DMS', yerr=BOLD_v1_dms_std )
rects_v1_ctl = ax.bar(index + width, BOLD_v1_ctl_avg, width, color='yellow', edgecolor='black', hatch='//',
                      label='V1 during CTL', yerr=BOLD_v1_dms_std, )

rects_v4_dms = ax.bar(index + width*2, BOLD_v4_dms_avg, width, color='green',
                      label='V4 during DMS', yerr=BOLD_v4_dms_std)
rects_v4_ctl = ax.bar(index + width*3, BOLD_v4_ctl_avg, width, color='green', edgecolor='black', hatch='//',
                      label='V4 during CTL', yerr=BOLD_v4_ctl_std)

rects_it_dms = ax.bar(index + width*4, BOLD_it_dms_avg, width, color='blue',
                      label='IT during DMS', yerr=BOLD_it_dms_std)
rects_it_ctl = ax.bar(index + width*5, BOLD_it_ctl_avg, width, color='blue', edgecolor='black', hatch='//',
                      label='IT during CTL', yerr=BOLD_it_ctl_std)

rects_fs_dms = ax.bar(index + width*6, BOLD_fs_dms_avg, width, color='orange',
                      label='FS during DMS', yerr=BOLD_fs_dms_std)
rects_fs_ctl = ax.bar(index + width*7, BOLD_fs_ctl_avg, width, color='orange', edgecolor='black', hatch='//',
                      label='FS during CTL', yerr=BOLD_fs_ctl_std)

rects_d1_dms = ax.bar(index + width*8, BOLD_d1_dms_avg, width, color='red',
                      label='D1 during DMS', yerr=BOLD_d1_dms_std)
rects_d1_ctl = ax.bar(index + width*9, BOLD_d1_ctl_avg, width, color='red', edgecolor='black', hatch='//',
                      label='D1 during CTL', yerr=BOLD_d1_ctl_std)

rects_d2_dms = ax.bar(index + width*10, BOLD_d2_dms_avg, width, color='pink',
                      label='D2 during DMS', yerr=BOLD_d2_dms_std)
rects_d2_ctl = ax.bar(index + width*11, BOLD_d2_ctl_avg, width, color='pink', edgecolor='black', hatch='//',
                      label='D2 during CTL', yerr=BOLD_d2_ctl_std)

rects_fr_dms = ax.bar(index + width*12, BOLD_fr_dms_avg, width, color='purple',
                      label='FR during DMS', yerr=BOLD_fr_dms_std)
rects_fr_ctl = ax.bar(index + width*13, BOLD_fr_ctl_avg, width, color='purple', edgecolor='black', hatch='//',
                      label='FR during CTL', yerr=BOLD_fr_ctl_std)

#ax.set_title('PSC ACROSS SUBJECTS IN ALL BRAIN REGIONS')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.set_ylabel('Signal change (%)')

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

