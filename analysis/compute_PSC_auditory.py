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
#   This file (compute_PSC_auditory.py) was created on April 1, 2016.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on April 26, 2016  
# **************************************************************************/

# compute_PSC_auditory.py
#
# Calculate and plot the Percent Signal Change (PSC) of fMRI BOLD timeseries across
# subjects for 2 conditions: TC-DMS and Tones-DMS.
#
# The inputs are BOLD timeseries for each subject, one timeseries per brain module.
# We take each timeseries and convert
# each to PSC by dividing every time point by the baseline (the mean of the PSL
# time course for a given subject and brain area), then multiplying by 100
# (Source: Percent Signal Change FAQ at mindhive.mit.edu). That way,
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
experiment_length = 980
trial_length = 80
num_of_trials = 12             # number of trials per subject

num_of_fmri_blocks = 1             # how many blocks of trials in the experiment
num_of_syn_blocks = 1              # we have more synaptic blocks than fmri blocks
                                    # because we get rid of blocks in BOLD timeseries

num_of_subjects = 1

scans_removed = 0             # number of scans removed from BOLD computation
synaptic_steps_removed = 0    # number of synaptic steps removed from synaptic
                              # computation
                              
num_of_modules = 4            # A1, A2, ST, PFC

# define total number of trials across subjects
total_trials = num_of_trials * num_of_subjects

# define total number of blocks across subjects
total_fmri_blocks = num_of_fmri_blocks * num_of_subjects
#total_syn_blocks  = num_of_syn_blocks * num_of_subjects

synaptic_timesteps = experiment_length - synaptic_steps_removed

# define the names of the input files where the BOLD timeseries are contained, for four conditions:
# TC-PSL, Tones-PSL, TC-DMS and Tones-DMS
BOLD_TC_PSL_subj = np.array(['subject_original/output.TC_PSL/lsnm_bold_balloon.npy'])
#                             'subject_2_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_3_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_4_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_5_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_6_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_7_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_8_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_9_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy',
#                             'subject_10_with_feedback/output.TC_PSL/lsnm_bold_balloon.npy'])

BOLD_Tones_PSL_subj = np.array(['subject_original_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy'])
#                             'subject_2_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_3_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_4_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_5_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_6_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_7_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_8_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_9_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy',
#                             'subject_10_with_feedback/output.Tones_PSL/lsnm_bold_balloon.npy'])

BOLD_TC_DMS_subj = np.array(['subject_original_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy'])
#                             'subject_2_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_3_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_4_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_5_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_6_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_7_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_8_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_9_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy',
#                             'subject_10_with_feedback/output.TC_DMS/lsnm_bold_balloon.npy'])

BOLD_Tones_DMS_subj = np.array(['subject_original_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy'])
#                                'subject_2_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_3_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_4_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_5_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_6_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_7_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_8_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_9_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy',
#                                'subject_10_with_feedback/output.Tones_DMS/lsnm_bold_balloon.npy'])

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# define output file where means, standard deviations, and variances will be stored
fc_stats_FILE = 'fc_stats.txt'

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 3.5 ms.
Ti = 0.0035 * 10

# Total time of scanning experiment in seconds (timesteps X 3.5)
T = 34.3

# Time for one complete trial in seconds
Ttrial = 4

# the scanning happened every Tr interval below (in seconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

num_of_scans = m.ceil(T / Tr - scans_removed)

print 'NUM OF SCANS', num_of_scans
                      
# Construct a numpy array of time points
time_in_seconds = np.arange(0, T, Tr)

scanning_timescale = np.arange(0, synaptic_timesteps, synaptic_timesteps / (T/Tr))

# open files containing BOLD time-series for four conditions: TC-PSL, Tones-PSL, TC-DMS and Tones-DMS
BOLD_TC_PSL = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
for idx in range(0, num_of_subjects):
    BOLD_TC_PSL[idx]  =  np.load(BOLD_TC_PSL_subj[idx])
BOLD_Tones_PSL = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
for idx in range(0, num_of_subjects):
    BOLD_Tones_PSL[idx]= np.load(BOLD_Tones_PSL_subj[idx])
BOLD_TC_DMS = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
for idx in range(0, num_of_subjects):
    BOLD_TC_DMS[idx]  =  np.load(BOLD_TC_DMS_subj[idx])
BOLD_Tones_DMS = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
for idx in range(0, num_of_subjects):
    BOLD_Tones_DMS[idx]= np.load(BOLD_Tones_DMS_subj[idx])

print BOLD_TC_PSL.shape

# Perform time-course normalization by converting each timeseries to percent signal change
# for each subject and for each module (TC and Tones DMS relative to the mean of the Tones PSL
# condition, as per Husain et al (2004), pp 1711, "Simulating PET and fMRI".
for s in range(0, num_of_subjects):
    for m in range(0, num_of_modules):
        timecourse_mean = np.mean(BOLD_Tones_DMS[s,m])
        BOLD_TC_DMS[s,m] = BOLD_TC_DMS[s,m] / timecourse_mean * 100. - 100.

#for s in range(0, num_of_subjects):
#    for m in range(0, num_of_modules):
#        timecourse_mean = np.mean(BOLD_Tones_PSL[s,m])
#        BOLD_Tones_DMS[s,m] = BOLD_Tones_DMS[s,m] / timecourse_mean * 100. - 100.

# now calculate PSC of TC-DMS with respect ot Tones-DMS
BOLD_DMS = BOLD_TC_DMS #- BOLD_Tones_DMS
        
mean_PSC_dms = np.zeros((num_of_subjects, num_of_modules))

# now, perform a mean of PSCs
for s in range(0, num_of_subjects):
    for m in range(0, num_of_modules):
        mean_PSC_dms[s,m] = np.mean(BOLD_DMS[s,m][5:10])

# now, calculate the average PSC across subjects for each brain region and for each condition:
BOLD_a1_dms_avg = np.mean(mean_PSC_dms[:,0]) 
BOLD_a2_dms_avg = np.mean(mean_PSC_dms[:,1])
BOLD_st_dms_avg = np.mean(mean_PSC_dms[:,2])
BOLD_pf_dms_avg = np.mean(mean_PSC_dms[:,3])

# calculate the variance as well, across subjects, for each brain regions and for each condition:
BOLD_a1_dms_var = np.var(mean_PSC_dms[:,0])
BOLD_a2_dms_var = np.var(mean_PSC_dms[:,1])
BOLD_st_dms_var = np.var(mean_PSC_dms[:,2])
BOLD_pf_dms_var = np.var(mean_PSC_dms[:,3])
#BOLD_v1_ctl_var = np.var(mean_PSC_ctl[:,0])
#BOLD_v4_ctl_var = np.var(mean_PSC_ctl[:,1])
#BOLD_it_ctl_var = np.var(mean_PSC_ctl[:,2])
#BOLD_pf_ctl_var = np.var(mean_PSC_ctl[:,3])
        
# calculate the standard deviation of the PSC across subjects, for each brain region and for each
# condition
BOLD_a1_dms_std = np.std(mean_PSC_dms[:,0])
BOLD_a2_dms_std = np.std(mean_PSC_dms[:,1])
BOLD_st_dms_std = np.std(mean_PSC_dms[:,2])
BOLD_pf_dms_std = np.std(mean_PSC_dms[:,3])
#BOLD_v1_ctl_std = np.std(mean_PSC_ctl[:,0])
#BOLD_v4_ctl_std = np.std(mean_PSC_ctl[:,1])
#BOLD_it_ctl_std = np.std(mean_PSC_ctl[:,2])
#BOLD_pf_ctl_std = np.std(mean_PSC_ctl[:,3])

# Display all PSCs, which represent the average of
# the Percent Signal Change per each brain area per each module of TC
# with respect to Tones
print 'BOLD A1 DMS Avg: ', BOLD_a1_dms_avg
print 'BOLD A2 DMS Avg: ', BOLD_a2_dms_avg
print 'BOLD ST DMS Avg: ', BOLD_st_dms_avg
print 'BOLD PF DMS Avg: ', BOLD_pf_dms_avg

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
#BOLD_v1_diff = BOLD_v1_dms_avg - BOLD_v1_ctl_avg
#BOLD_v4_diff = BOLD_v4_dms_avg - BOLD_v4_ctl_avg
#BOLD_it_diff = BOLD_it_dms_avg - BOLD_it_ctl_avg
#BOLD_fs_diff = BOLD_fs_dms_avg - BOLD_fs_ctl_avg
#BOLD_d1_diff = BOLD_d1_dms_avg - BOLD_d1_ctl_avg
#BOLD_d2_diff = BOLD_d2_dms_avg - BOLD_d2_ctl_avg
#BOLD_fr_diff = BOLD_fr_dms_avg - BOLD_fr_ctl_avg
#     (2) Calculate, for both control and DMS, the variance divided by sample size minus 1:
#BOLD_v1_ctl_a= BOLD_v1_ctl_var / (sample_size-1)
#BOLD_v4_ctl_a= BOLD_v4_ctl_var / (sample_size-1)
#BOLD_it_ctl_a= BOLD_it_ctl_var / (sample_size-1)
#BOLD_fs_ctl_a= BOLD_fs_ctl_var / (sample_size-1)
#BOLD_d1_ctl_a= BOLD_d1_ctl_var / (sample_size-1)
#BOLD_d2_ctl_a= BOLD_d2_ctl_var / (sample_size-1)
#BOLD_fr_ctl_a= BOLD_fr_ctl_var / (sample_size-1)
#BOLD_v1_dms_a= BOLD_v1_dms_var / (sample_size-1)
#BOLD_v4_dms_a= BOLD_v4_dms_var / (sample_size-1)
#BOLD_it_dms_a= BOLD_it_dms_var / (sample_size-1)
#BOLD_fs_dms_a= BOLD_fs_dms_var / (sample_size-1)
#BOLD_d1_dms_a= BOLD_d1_dms_var / (sample_size-1)
#BOLD_d2_dms_a= BOLD_d2_dms_var / (sample_size-1)
#BOLD_fr_dms_a= BOLD_fr_dms_var / (sample_size-1)
#     (3) Add results obtained for CTL and DMS in step (2) together:
#BOLD_v1_a = BOLD_v1_ctl_a + BOLD_v1_dms_a
#BOLD_v4_a = BOLD_v4_ctl_a + BOLD_v4_dms_a
#BOLD_it_a = BOLD_it_ctl_a + BOLD_it_dms_a
#BOLD_fs_a = BOLD_fs_ctl_a + BOLD_fs_dms_a
#BOLD_d1_a = BOLD_d1_ctl_a + BOLD_d1_dms_a
#BOLD_d2_a = BOLD_d2_ctl_a + BOLD_d2_dms_a
#BOLD_fr_a = BOLD_fr_ctl_a + BOLD_fr_dms_a
#     (4) Take the square root the results in step (3):
#sqrt_BOLD_v1_a= np.sqrt(BOLD_v1_a)
#sqrt_BOLD_v4_a= np.sqrt(BOLD_v4_a)
#sqrt_BOLD_it_a= np.sqrt(BOLD_it_a)
#sqrt_BOLD_fs_a= np.sqrt(BOLD_fs_a)
#sqrt_BOLD_d1_a= np.sqrt(BOLD_d1_a)
#sqrt_BOLD_d2_a= np.sqrt(BOLD_d2_a)
#sqrt_BOLD_fr_a= np.sqrt(BOLD_fr_a)
#     (5) Divide the results of step (1) by the results of step (4) to obtain 't':
#BOLD_v1_t= BOLD_v1_diff / sqrt_BOLD_v1_a
#BOLD_v4_t= BOLD_v4_diff / sqrt_BOLD_v4_a
#BOLD_it_t= BOLD_it_diff / sqrt_BOLD_it_a
#BOLD_fs_t= BOLD_fs_diff / sqrt_BOLD_fs_a
#BOLD_d1_t= BOLD_d1_diff / sqrt_BOLD_d1_a
#BOLD_d2_t= BOLD_d2_diff / sqrt_BOLD_d2_a
#BOLD_fr_t= BOLD_fr_diff / sqrt_BOLD_fr_a
#     (6) Calculate the degrees of freedom (add up number of observations for each group
#         minus number of groups):
#dof = sample_size + sample_size - number_of_groups
#print 'Degrees of freedom: ', dof
#     (7) find the p-values for the above 't' and 'degrees of freedom':
#BOLD_v1_p_values = t.sf(BOLD_v1_t, dof)
#BOLD_v4_p_values = t.sf(BOLD_v4_t, dof)
#BOLD_it_p_values = t.sf(BOLD_it_t, dof)
#BOLD_fs_p_values = t.sf(BOLD_fs_t, dof)
#BOLD_d1_p_values = t.sf(BOLD_d1_t, dof)
#BOLD_d2_p_values = t.sf(BOLD_d2_t, dof)
#BOLD_fr_p_values = t.sf(BOLD_fr_t, dof)

#print 't-value for v1 BOLD signal difference: ', BOLD_v1_t
#print 't-value for v4 BOLD signal difference: ', BOLD_v4_t
#print 't-value for it BOLD signal difference: ', BOLD_it_t
#print 't-value for fs BOLD signal difference: ', BOLD_fs_t
#print 't-value for d1 BOLD signal difference: ', BOLD_d1_t
#print 't-value for d2 BOLD signal difference: ', BOLD_d2_t
#print 't-value for fr BOLD signal difference: ', BOLD_fr_t


# increase font size prior to plotting
plt.rcParams.update({'font.size': 15})

# define number of groups to plot
N = 1

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.2                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([0,100])

# now, group the values to be plotted by brain module and by task condition

rects_a1_dms = ax.bar(index, BOLD_a1_dms_avg, width, color='yellow',
                      label='A1', yerr=BOLD_a1_dms_std)
#rects_v1_ctl = ax.bar(index + width, BOLD_v1_ctl_avg, width, color='yellow', edgecolor='black', hatch='//',
#                      label='V1 during CTL', yerr=BOLD_v1_dms_std, )

rects_a2_dms = ax.bar(index + width*2, BOLD_a2_dms_avg, width, color='green',
                      label='A2', yerr=BOLD_a2_dms_std)
#rects_v4_ctl = ax.bar(index + width*3, BOLD_v4_ctl_avg, width, color='green', edgecolor='black', hatch='//',
#                      label='V4 during CTL', yerr=BOLD_v4_ctl_std)

rects_st_dms = ax.bar(index + width*4, BOLD_st_dms_avg, width, color='blue',
                      label='ST', yerr=BOLD_st_dms_std)
#rects_it_ctl = ax.bar(index + width*5, BOLD_it_ctl_avg, width, color='blue', edgecolor='black', hatch='//',
#                      label='IT during CTL', yerr=BOLD_it_ctl_std)

rects_pf_dms = ax.bar(index + width*6, BOLD_pf_dms_avg, width, color='red',
                      label='PFC', yerr=BOLD_st_dms_std)
#rects_pf_ctl = ax.bar(index + width*7, BOLD_fs_ctl_avg, width, color='orange', edgecolor='black', hatch='//',
#                      label='FS during CTL', yerr=BOLD_fs_ctl_std)

#ax.set_title('PSC ACROSS SUBJECTS IN ALL BRAIN REGIONS')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.set_ylabel('Percent Signal Change (%)')

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

# Show the plots on the screen
plt.show()

