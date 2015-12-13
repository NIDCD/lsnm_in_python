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
#   Last updated by Antonio Ulloa on December 2, 2015  
# **************************************************************************/

# compute_PSC.py
#
# Calculate and plot the Percent Signal Change (PSC) of fMRI BOLD timeseries across
# subjects for 2 conditions: DMS and passive viewing.
#
# The inputs are BOLD timeseries for each subject. We take each timeseries and convert
# each to PSC by dividing every time point by the baseline (the mean of the whole
# time course), then multiplying by 100 and resting 100. That way, we obtain a
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
BOLD_ts_subj=np.array(['subject_11/output.36trials/lsnm_bold_balloon.npy',
                       'subject_12/output.36trials/lsnm_bold_balloon.npy',
                       'subject_13/output.36trials/lsnm_bold_balloon.npy',
                       'subject_14/output.36trials/lsnm_bold_balloon.npy',
                       'subject_15/output.36trials/lsnm_bold_balloon.npy',
                       'subject_16/output.36trials/lsnm_bold_balloon.npy',
                       'subject_17/output.36trials/lsnm_bold_balloon.npy',
                       'subject_18/output.36trials/lsnm_bold_balloon.npy',
                       'subject_19/output.36trials/lsnm_bold_balloon.npy',
                       'subject_20/output.36trials/lsnm_bold_balloon.npy'])

#syn_ts_subj=np.array(['subject_11/output.36trials/synaptic_in_ROI.npy',
#                       'subject_12/output.36trials/synaptic_in_ROI.npy',
#                       'subject_13/output.36trials/synaptic_in_ROI.npy',
#                       'subject_14/output.36trials/synaptic_in_ROI.npy',
#                       'subject_15/output.36trials/synaptic_in_ROI.npy',
#                       'subject_16/output.36trials/synaptic_in_ROI.npy',
#                       'subject_17/output.36trials/synaptic_in_ROI.npy',
#                       'subject_18/output.36trials/synaptic_in_ROI.npy',
#                       'subject_19/output.36trials/synaptic_in_ROI.npy',
#                       'subject_20/output.36trials/synaptic_in_ROI.npy'])

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
#synaptic_timescale = np.arange(0, synaptic_timesteps)

# open files containing BOLD time-series
BOLD_ts = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
#syn_ts = np.zeros((num_of_subjects, num_of_modules, synaptic_timesteps))
for idx in range(0, num_of_subjects):
#    syn_ts[idx] = np.load(syn_ts_subj[idx])
    BOLD_ts[idx] = np.load(BOLD_ts_subj[idx])    

# Concatenate the time-series of each module together, across all subjects:
#syn_ts = np.reshape(syn_ts, (num_of_modules, num_of_subjects*synaptic_timesteps))
BOLD_ts = np.reshape(BOLD_ts, (num_of_modules, num_of_subjects*num_of_scans))

# now, split all time-series in blocks
#syn_ts_v1=np.array_split(syn_ts[0], total_syn_blocks)
#syn_ts_v4=np.array_split(syn_ts[1], total_syn_blocks)
#syn_ts_it=np.array_split(syn_ts[2], total_syn_blocks)
#syn_ts_fs=np.array_split(syn_ts[3], total_syn_blocks)
#syn_ts_d1=np.array_split(syn_ts[4], total_syn_blocks)
#syn_ts_d2=np.array_split(syn_ts[5], total_syn_blocks)
#syn_ts_fr=np.array_split(syn_ts[6], total_syn_blocks)
BOLD_ts_v1=np.array_split(BOLD_ts[0], total_fmri_blocks)
BOLD_ts_v4=np.array_split(BOLD_ts[1], total_fmri_blocks)
BOLD_ts_it=np.array_split(BOLD_ts[2], total_fmri_blocks)
BOLD_ts_fs=np.array_split(BOLD_ts[3], total_fmri_blocks)
BOLD_ts_d1=np.array_split(BOLD_ts[4], total_fmri_blocks)
BOLD_ts_d2=np.array_split(BOLD_ts[5], total_fmri_blocks)
BOLD_ts_fr=np.array_split(BOLD_ts[6], total_fmri_blocks)
BOLD_ts_lit=np.array_split(BOLD_ts[7], total_fmri_blocks)

# define an array with location of control blocks, and another array
# with location of task (DMS) blocks, relative to
# an array that contains all blocks (task-related blocks included)
#syn_control_block_ids = np.arange(1, total_syn_blocks, 2)
#syn_dms_block_ids =     np.arange(0, total_syn_blocks, 2)
fmri_control_block_ids = np.arange(1, total_fmri_blocks, 2)
fmri_dms_block_ids =     np.arange(0, total_fmri_blocks, 2)

# now, create an array of BOLD time-series containing DMS trials only: 
#syn_v1_dms_blocks = np.delete(np.asarray(syn_ts_v1), syn_control_block_ids, axis=0)
#syn_v4_dms_blocks = np.delete(np.asarray(syn_ts_v4), syn_control_block_ids, axis=0)
#syn_it_dms_blocks = np.delete(np.asarray(syn_ts_it), syn_control_block_ids, axis=0)
#syn_fs_dms_blocks = np.delete(np.asarray(syn_ts_fs), syn_control_block_ids, axis=0)
#syn_d1_dms_blocks = np.delete(np.asarray(syn_ts_d1), syn_control_block_ids, axis=0)
#syn_d2_dms_blocks = np.delete(np.asarray(syn_ts_d2), syn_control_block_ids, axis=0)
#syn_fr_dms_blocks = np.delete(np.asarray(syn_ts_fr), syn_control_block_ids, axis=0)
BOLD_v1_dms_blocks = np.delete(np.asarray(BOLD_ts_v1), fmri_control_block_ids, axis=0)
BOLD_v4_dms_blocks = np.delete(np.asarray(BOLD_ts_v4), fmri_control_block_ids, axis=0)
BOLD_it_dms_blocks = np.delete(np.asarray(BOLD_ts_it), fmri_control_block_ids, axis=0)
BOLD_fs_dms_blocks = np.delete(np.asarray(BOLD_ts_fs), fmri_control_block_ids, axis=0)
BOLD_d1_dms_blocks = np.delete(np.asarray(BOLD_ts_d1), fmri_control_block_ids, axis=0)
BOLD_d2_dms_blocks = np.delete(np.asarray(BOLD_ts_d2), fmri_control_block_ids, axis=0)
BOLD_fr_dms_blocks = np.delete(np.asarray(BOLD_ts_fr), fmri_control_block_ids, axis=0)
BOLD_lit_dms_blocks = np.delete(np.asarray(BOLD_ts_lit), fmri_control_block_ids, axis=0)

# ... and concatenate those DMS BOLD timeseries together
#syn_v1_dms_ts = np.concatenate(syn_v1_dms_blocks)
#syn_v4_dms_ts = np.concatenate(syn_v4_dms_blocks)
#syn_it_dms_ts = np.concatenate(syn_it_dms_blocks)
#syn_fs_dms_ts = np.concatenate(syn_fs_dms_blocks)
#syn_d1_dms_ts = np.concatenate(syn_d1_dms_blocks)
#syn_d2_dms_ts = np.concatenate(syn_d2_dms_blocks)
#syn_fr_dms_ts = np.concatenate(syn_fr_dms_blocks)
BOLD_v1_dms_ts = np.concatenate(BOLD_v1_dms_blocks)
BOLD_v4_dms_ts = np.concatenate(BOLD_v4_dms_blocks)
BOLD_it_dms_ts = np.concatenate(BOLD_it_dms_blocks)
BOLD_fs_dms_ts = np.concatenate(BOLD_fs_dms_blocks)
BOLD_d1_dms_ts = np.concatenate(BOLD_d1_dms_blocks)
BOLD_d2_dms_ts = np.concatenate(BOLD_d2_dms_blocks)
BOLD_fr_dms_ts = np.concatenate(BOLD_fr_dms_blocks)
BOLD_lit_dms_ts = np.concatenate(BOLD_lit_dms_blocks)

# but also, get rid of the DMS blocks, to create arrays with only control trials
#syn_v1_control_blocks = np.delete(np.asarray(syn_ts_v1), syn_dms_block_ids, axis=0)
#syn_v4_control_blocks = np.delete(np.asarray(syn_ts_v4), syn_dms_block_ids, axis=0)
#syn_it_control_blocks = np.delete(np.asarray(syn_ts_it), syn_dms_block_ids, axis=0)
#syn_fs_control_blocks = np.delete(np.asarray(syn_ts_fs), syn_dms_block_ids, axis=0)
#syn_d1_control_blocks = np.delete(np.asarray(syn_ts_d1), syn_dms_block_ids, axis=0)
#syn_d2_control_blocks = np.delete(np.asarray(syn_ts_d2), syn_dms_block_ids, axis=0)
#syn_fr_control_blocks = np.delete(np.asarray(syn_ts_fr), syn_dms_block_ids, axis=0)
BOLD_v1_control_blocks = np.delete(np.asarray(BOLD_ts_v1), fmri_dms_block_ids, axis=0)
BOLD_v4_control_blocks = np.delete(np.asarray(BOLD_ts_v4), fmri_dms_block_ids, axis=0)
BOLD_it_control_blocks = np.delete(np.asarray(BOLD_ts_it), fmri_dms_block_ids, axis=0)
BOLD_fs_control_blocks = np.delete(np.asarray(BOLD_ts_fs), fmri_dms_block_ids, axis=0)
BOLD_d1_control_blocks = np.delete(np.asarray(BOLD_ts_d1), fmri_dms_block_ids, axis=0)
BOLD_d2_control_blocks = np.delete(np.asarray(BOLD_ts_d2), fmri_dms_block_ids, axis=0)
BOLD_fr_control_blocks = np.delete(np.asarray(BOLD_ts_fr), fmri_dms_block_ids, axis=0)
BOLD_lit_control_blocks = np.delete(np.asarray(BOLD_ts_lit), fmri_dms_block_ids, axis=0)

# ... and concatenate the control blocks together
#syn_v1_ctl_ts = np.concatenate(syn_v1_control_blocks)
#syn_v4_ctl_ts = np.concatenate(syn_v4_control_blocks)
#syn_it_ctl_ts = np.concatenate(syn_it_control_blocks)
#syn_fs_ctl_ts = np.concatenate(syn_fs_control_blocks)
#syn_d1_ctl_ts = np.concatenate(syn_d1_control_blocks)
#syn_d2_ctl_ts = np.concatenate(syn_d2_control_blocks)
#syn_fr_ctl_ts = np.concatenate(syn_fr_control_blocks)
BOLD_v1_ctl_ts = np.concatenate(BOLD_v1_control_blocks)
BOLD_v4_ctl_ts = np.concatenate(BOLD_v4_control_blocks)
BOLD_it_ctl_ts = np.concatenate(BOLD_it_control_blocks)
BOLD_fs_ctl_ts = np.concatenate(BOLD_fs_control_blocks)
BOLD_d1_ctl_ts = np.concatenate(BOLD_d1_control_blocks)
BOLD_d2_ctl_ts = np.concatenate(BOLD_d2_control_blocks)
BOLD_fr_ctl_ts = np.concatenate(BOLD_fr_control_blocks)
BOLD_lit_ctl_ts = np.concatenate(BOLD_lit_control_blocks)

# Average all timepoints together for the DMS condition:
#syn_v1_dms_PSC = np.mean(syn_v1_dms_ts)
#syn_v4_dms_PSC = np.mean(syn_v4_dms_ts)
#syn_it_dms_PSC = np.mean(syn_it_dms_ts)
#syn_fs_dms_PSC = np.mean(syn_fs_dms_ts)
#syn_d1_dms_PSC = np.mean(syn_d1_dms_ts)
#syn_d2_dms_PSC = np.mean(syn_d2_dms_ts)
#syn_fr_dms_PSC = np.mean(syn_fr_dms_ts)
BOLD_v1_dms_PSC = np.mean(BOLD_v1_dms_ts)
BOLD_v4_dms_PSC = np.mean(BOLD_v4_dms_ts)
BOLD_it_dms_PSC = np.mean(BOLD_it_dms_ts)
BOLD_fs_dms_PSC = np.mean(BOLD_fs_dms_ts)
BOLD_d1_dms_PSC = np.mean(BOLD_d1_dms_ts)
BOLD_d2_dms_PSC = np.mean(BOLD_d2_dms_ts)
BOLD_fr_dms_PSC = np.mean(BOLD_fr_dms_ts)
BOLD_lit_dms_PSC = np.mean(BOLD_lit_dms_ts)

# Average all timepoints together for the control condition:
#syn_v1_ctl_PSC = np.mean(syn_v1_ctl_ts)
#syn_v4_ctl_PSC = np.mean(syn_v4_ctl_ts)
#syn_it_ctl_PSC = np.mean(syn_it_ctl_ts)
#syn_fs_ctl_PSC = np.mean(syn_fs_ctl_ts)
#syn_d1_ctl_PSC = np.mean(syn_d1_ctl_ts)
#syn_d2_ctl_PSC = np.mean(syn_d2_ctl_ts)
#syn_fr_ctl_PSC = np.mean(syn_fr_ctl_ts)
BOLD_v1_ctl_PSC = np.mean(BOLD_v1_ctl_ts)
BOLD_v4_ctl_PSC = np.mean(BOLD_v4_ctl_ts)
BOLD_it_ctl_PSC = np.mean(BOLD_it_ctl_ts)
BOLD_fs_ctl_PSC = np.mean(BOLD_fs_ctl_ts)
BOLD_d1_ctl_PSC = np.mean(BOLD_d1_ctl_ts)
BOLD_d2_ctl_PSC = np.mean(BOLD_d2_ctl_ts)
BOLD_fr_ctl_PSC = np.mean(BOLD_fr_ctl_ts)
BOLD_lit_ctl_PSC = np.mean(BOLD_lit_ctl_ts)

# Normalize (convert to percent signal change) each DMS timeseries first
# per subject and per module relative to the control condition
#syn_v1_dms_PSC = (syn_v1_dms_PSC - syn_v1_ctl_PSC) / syn_v1_ctl_PSC * 100.
#syn_v4_dms_PSC = (syn_v4_dms_PSC - syn_v4_ctl_PSC) / syn_v4_ctl_PSC * 100.
#syn_it_dms_PSC = (syn_it_dms_PSC - syn_it_ctl_PSC) / syn_it_ctl_PSC * 100.
#syn_fs_dms_PSC = (syn_fs_dms_PSC - syn_fs_ctl_PSC) / syn_fs_ctl_PSC * 100.
#syn_d1_dms_PSC = (syn_d1_dms_PSC - syn_d1_ctl_PSC) / syn_d1_ctl_PSC * 100.
#syn_d2_dms_PSC = (syn_d2_dms_PSC - syn_d2_ctl_PSC) / syn_d2_ctl_PSC * 100.
#syn_fr_dms_PSC = (syn_fr_dms_PSC - syn_fr_ctl_PSC) / syn_fr_ctl_PSC * 100.
BOLD_v1_dms_PSC = (BOLD_v1_dms_PSC - BOLD_v1_ctl_PSC) / BOLD_v1_ctl_PSC * 100.
BOLD_v4_dms_PSC = (BOLD_v4_dms_PSC - BOLD_v4_ctl_PSC) / BOLD_v4_ctl_PSC * 100.
BOLD_it_dms_PSC = (BOLD_it_dms_PSC - BOLD_it_ctl_PSC) / BOLD_it_ctl_PSC * 100.
BOLD_fs_dms_PSC = (BOLD_fs_dms_PSC - BOLD_fs_ctl_PSC) / BOLD_fs_ctl_PSC * 100.
BOLD_d1_dms_PSC = (BOLD_d1_dms_PSC - BOLD_d1_ctl_PSC) / BOLD_d1_ctl_PSC * 100.
BOLD_d2_dms_PSC = (BOLD_d2_dms_PSC - BOLD_d2_ctl_PSC) / BOLD_d2_ctl_PSC * 100.
BOLD_fr_dms_PSC = (BOLD_fr_dms_PSC - BOLD_fr_ctl_PSC) / BOLD_fr_ctl_PSC * 100.
BOLD_lit_dms_PSC = (BOLD_lit_dms_PSC - BOLD_lit_ctl_PSC) / BOLD_lit_ctl_PSC * 100.


# Display all PSC values:
#print 'SYN V1 PSC: ', syn_v1_dms_PSC
#print 'SYN V4 PSC: ', syn_v4_dms_PSC
#print 'SYN IT PSC: ', syn_it_dms_PSC
#print 'SYN FS PSC: ', syn_fs_dms_PSC
#print 'SYN D1 PSC: ', syn_d1_dms_PSC
#print 'SYN D2 PSC: ', syn_d2_dms_PSC
#print 'SYN FR PSC: ', syn_fr_dms_PSC

print 'BOLD V1 PSC: ', BOLD_v1_dms_PSC
print 'BOLD V4 PSC: ', BOLD_v4_dms_PSC
print 'BOLD IT PSC: ', BOLD_it_dms_PSC
print 'BOLD FS PSC: ', BOLD_fs_dms_PSC
print 'BOLD D1 PSC: ', BOLD_d1_dms_PSC
print 'BOLD D2 PSC: ', BOLD_d2_dms_PSC
print 'BOLD FR PSC: ', BOLD_fr_dms_PSC
print 'BOLD cIT PSC: ', BOLD_lit_dms_PSC

# define number of groups to plot
N = 1

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([0,3.5])

# now, group the values to be plotted by brain module
#v1_psc = (syn_v1_dms_PSC, BOLD_v1_dms_PSC)
#v4_psc = (syn_v4_dms_PSC, BOLD_v4_dms_PSC)
#it_psc = (syn_it_dms_PSC, BOLD_it_dms_PSC)
#fs_psc = (syn_fs_dms_PSC, BOLD_fs_dms_PSC)
#d1_psc = (syn_d1_dms_PSC, BOLD_d1_dms_PSC)
#d2_psc = (syn_d2_dms_PSC, BOLD_d2_dms_PSC)
#fr_psc = (syn_fr_dms_PSC, BOLD_fr_dms_PSC)

rects_v1 = ax.bar(index, BOLD_v1_dms_PSC, width, color='yellow', label='V1')

rects_v4 = ax.bar(index + width, BOLD_v4_dms_PSC, width, color='green', label='V4')

rects_it = ax.bar(index + width*2, BOLD_it_dms_PSC, width, color='blue', label='IT')

rects_fs = ax.bar(index + width*3, BOLD_fs_dms_PSC, width, color='orange', label='FS')

rects_d1 = ax.bar(index + width*4, BOLD_d1_dms_PSC, width, color='red', label='D1')

rects_d2 = ax.bar(index + width*5, BOLD_d2_dms_PSC, width, color='pink', label='D2')

rects_fr = ax.bar(index + width*6, BOLD_fr_dms_PSC, width, color='purple', label='FR')

rects_lit= ax.bar(index + width*7, BOLD_lit_dms_PSC, width, color='lightblue', label='cIT')

#ax.set_title('PSC ACROSS SUBJECTS IN ALL BRAIN REGIONS')

# get rid of x axis ticks and labels
ax.set_xticks([])

#ax.set_xlabel('SYNAPTIC PSC                                        fMRI PSC')
#ax.xaxis.set_label_coords(0.5, -0.025)

ax.set_ylabel('Percent signal change')

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

# Show the plots on the screen
plt.show()

