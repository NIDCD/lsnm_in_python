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
#   This file (func_conn_fmri_visual.py) was created on July 10, 2015.
#
#   Based in part on Matlab scripts by Horwitz et al.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 13, 2015  
# **************************************************************************/

# func_conn_fmri_visual.py
#
# Calculate and plot functional connectivity (within-task time series correlation)
# of the BOLD timeseries in IT with the BOLD timeseries of all other simulated brain
# regions.

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import math as m

from scipy.stats import poisson

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110
number_of_trials = 36

num_of_fmri_blocks = 12

blocks_removed = 1

scans_removed = 8

trials_removed = 3

trials = number_of_trials - trials_removed

fmri_blocks = num_of_fmri_blocks - blocks_removed

synaptic_timesteps = experiment_length

# define the name of the input file where the BOLD timeseries are stored
BOLD_file = 'lsnm_bold_balloon.npy'

# define the name of the output file where the functional connectivity timeseries will be stored
func_conn_dms_file = 'corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_ctl_file = 'corr_fmri_IT_vs_all_ctl_balloon.npy'

# define an array with location of control trials, and another array
# with location of task-related trials, relative to
# an array that contains all trials (task-related trials included)
# because we lost 3 trials at the beginning of the fmri experiment, we
#  were left with only 33 trials,
control_trials = np.array([1,2,3,7,8,9,13,14,15,19,20,21,25,26,27,31,32,33])
dms_trials =     np.array([0,4,5,6,10,11,12,16,17,18,22,23,24,28,29,30])

# load fMRI BOLD time-series into an array
BOLD = np.load(BOLD_file)

# extract each ROI's BOLD time-series from array
v1_BOLD = BOLD[0]
v4_BOLD = BOLD[1]
it_BOLD = BOLD[2]
fs_BOLD = BOLD[3]
d1_BOLD = BOLD[4]
d2_BOLD = BOLD[5]
fr_BOLD = BOLD[6]
lit_BOLD= BOLD[7]

# And now we scale it up 100 times its size, so that we can subdivide the arrays
# in equally sized subarrays
#x_100 = np.linspace(0, v1_BOLD.size-1, num=v1_BOLD.size*100.)
#scanning_timescale = np.linspace(0, v1_BOLD.size-1, v1_BOLD.size)
#v1_BOLD  = np.interp(x_100, scanning_timescale, v1_BOLD)
#v4_BOLD  = np.interp(x_100, scanning_timescale, v4_BOLD)
#it_BOLD  = np.interp(x_100, scanning_timescale, it_BOLD)
#fs_BOLD  = np.interp(x_100, scanning_timescale, fs_BOLD)
#d1_BOLD  = np.interp(x_100, scanning_timescale, d1_BOLD)
#d2_BOLD  = np.interp(x_100, scanning_timescale, d2_BOLD)
#fr_BOLD  = np.interp(x_100, scanning_timescale, fr_BOLD)
#lit_BOLD = np.interp(x_100, scanning_timescale, lit_BOLD)

# Get rid of the control trials in the BOLD signal arrays,
# by separating the task-related trials and concatenating them
# together. Remember that each trial is a number of synaptic timesteps
# long.

# first, split the arrays into subarrays, each one containing a single trial
it_subarrays = np.array_split(it_BOLD, trials)
v1_subarrays = np.array_split(v1_BOLD, trials)
v4_subarrays = np.array_split(v4_BOLD, trials)
fs_subarrays = np.array_split(fs_BOLD, trials)
d1_subarrays = np.array_split(d1_BOLD, trials)
d2_subarrays = np.array_split(d2_BOLD, trials)
fr_subarrays = np.array_split(fr_BOLD, trials)
lit_subarrays= np.array_split(lit_BOLD,trials)

# we get rid of the inter-trial interval for each and all trials (1 second at the
# end of each trial. 2 seconds = 1 scan.
#for idx in range(len(it_subarrays)):
#    it_subarrays[idx] = np.delete(it_subarrays[idx], -1)
#    v1_subarrays[idx] = np.delete(v1_subarrays[idx], -1)
#    v4_subarrays[idx] = np.delete(v4_subarrays[idx], -1)
#    fs_subarrays[idx] = np.delete(fs_subarrays[idx], -1)
#    d1_subarrays[idx] = np.delete(d1_subarrays[idx], -1)
#    d2_subarrays[idx] = np.delete(d2_subarrays[idx], -1)
#    fr_subarrays[idx] = np.delete(fr_subarrays[idx], -1)
#    lit_subarrays[idx]= np.delete(lit_subarrays[idx], -1)

# now, get rid of the control trials...
it_DMS_trials = np.delete(it_subarrays, control_trials, axis=0)
v1_DMS_trials = np.delete(v1_subarrays, control_trials, axis=0)
v4_DMS_trials = np.delete(v4_subarrays, control_trials, axis=0)
d1_DMS_trials = np.delete(d1_subarrays, control_trials, axis=0)
d2_DMS_trials = np.delete(d2_subarrays, control_trials, axis=0)
fs_DMS_trials = np.delete(fs_subarrays, control_trials, axis=0)
fr_DMS_trials = np.delete(fr_subarrays, control_trials, axis=0)
lit_DMS_trials= np.delete(lit_subarrays,control_trials, axis=0)

# ... and concatenate the task-related trials together
it_BOLD_dms = np.concatenate(it_DMS_trials)
v1_BOLD_dms = np.concatenate(v1_DMS_trials)
v4_BOLD_dms = np.concatenate(v4_DMS_trials)
d1_BOLD_dms = np.concatenate(d1_DMS_trials)
d2_BOLD_dms = np.concatenate(d2_DMS_trials)
fs_BOLD_dms = np.concatenate(fs_DMS_trials)
fr_BOLD_dms = np.concatenate(fr_DMS_trials)
lit_BOLD_dms= np.concatenate(lit_DMS_trials)

# but also, get rid of the DMS task trials, to create arrays with only control trials
v1_control_trials = np.delete(v1_subarrays, dms_trials, axis=0)
v4_control_trials = np.delete(v4_subarrays, dms_trials, axis=0)
it_control_trials = np.delete(it_subarrays, dms_trials, axis=0)
fs_control_trials = np.delete(fs_subarrays, dms_trials, axis=0)
d1_control_trials = np.delete(d1_subarrays, dms_trials, axis=0)
d2_control_trials = np.delete(d2_subarrays, dms_trials, axis=0)
fr_control_trials = np.delete(fr_subarrays, dms_trials, axis=0)
lit_control_trials= np.delete(lit_subarrays,dms_trials, axis=0)

# ... and concatenate the control task trials together
v1_BOLD_ctl = np.concatenate(v1_control_trials)
v4_BOLD_ctl = np.concatenate(v4_control_trials)
it_BOLD_ctl = np.concatenate(it_control_trials)
fs_BOLD_ctl = np.concatenate(fs_control_trials)
d1_BOLD_ctl = np.concatenate(d1_control_trials)
d2_BOLD_ctl = np.concatenate(d2_control_trials)
fr_BOLD_ctl = np.concatenate(fr_control_trials)
lit_BOLD_ctl= np.concatenate(lit_control_trials)

# now, convert DMS and control timeseries into pandas timeseries, so we can analyze it
IT_dms_ts = pd.Series(it_BOLD_dms)
V1_dms_ts = pd.Series(v1_BOLD_dms)
V4_dms_ts = pd.Series(v4_BOLD_dms)
D1_dms_ts = pd.Series(d1_BOLD_dms)
D2_dms_ts = pd.Series(d2_BOLD_dms)
FS_dms_ts = pd.Series(fs_BOLD_dms)
FR_dms_ts = pd.Series(fr_BOLD_dms)
LIT_dms_ts= pd.Series(lit_BOLD_dms)

IT_ctl_ts = pd.Series(it_BOLD_ctl)
V1_ctl_ts = pd.Series(v1_BOLD_ctl)
V4_ctl_ts = pd.Series(v4_BOLD_ctl)
D1_ctl_ts = pd.Series(d1_BOLD_ctl)
D2_ctl_ts = pd.Series(d2_BOLD_ctl)
FS_ctl_ts = pd.Series(fs_BOLD_ctl)
FR_ctl_ts = pd.Series(fr_BOLD_ctl)
LIT_ctl_ts= pd.Series(lit_BOLD_ctl)


# ... and calculate the functional connectivity of IT with the other modules
funct_conn_it_v1_dms = IT_dms_ts.corr(V1_dms_ts)
funct_conn_it_v4_dms = IT_dms_ts.corr(V4_dms_ts)
funct_conn_it_d1_dms = IT_dms_ts.corr(D1_dms_ts)
funct_conn_it_d2_dms = IT_dms_ts.corr(D2_dms_ts)
funct_conn_it_fs_dms = IT_dms_ts.corr(FS_dms_ts)
funct_conn_it_fr_dms = IT_dms_ts.corr(FR_dms_ts)
funct_conn_it_lit_dms= IT_dms_ts.corr(LIT_dms_ts)

funct_conn_it_v1_ctl = IT_ctl_ts.corr(V1_ctl_ts)
funct_conn_it_v4_ctl = IT_ctl_ts.corr(V4_ctl_ts)
funct_conn_it_d1_ctl = IT_ctl_ts.corr(D1_ctl_ts)
funct_conn_it_d2_ctl = IT_ctl_ts.corr(D2_ctl_ts)
funct_conn_it_fs_ctl = IT_ctl_ts.corr(FS_ctl_ts)
funct_conn_it_fr_ctl = IT_ctl_ts.corr(FR_ctl_ts)
funct_conn_it_lit_ctl= IT_ctl_ts.corr(LIT_ctl_ts)

func_conn_dms = np.array([funct_conn_it_v1_dms, funct_conn_it_v4_dms,
                          funct_conn_it_fs_dms, funct_conn_it_d1_dms,
                          funct_conn_it_d2_dms, funct_conn_it_fr_dms,
                          funct_conn_it_lit_dms ])

func_conn_ctl = np.array([funct_conn_it_v1_ctl,funct_conn_it_v4_ctl,
                          funct_conn_it_fs_ctl,funct_conn_it_d1_ctl,
                          funct_conn_it_d2_ctl,funct_conn_it_fr_ctl,
                          funct_conn_it_lit_ctl ])

# now, save all correlation coefficients to output files 
np.save(func_conn_dms_file, func_conn_dms)
np.save(func_conn_ctl_file, func_conn_ctl)

# define number of groups to plot
N = 2

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([0,1])

# now, group the values to be plotted by brain module
it_v1_corr = (funct_conn_it_v1_dms, funct_conn_it_v1_ctl)
it_v4_corr = (funct_conn_it_v4_dms, funct_conn_it_v4_ctl)
it_d1_corr = (funct_conn_it_d1_dms, funct_conn_it_d1_ctl)
it_d2_corr = (funct_conn_it_d2_dms, funct_conn_it_d2_ctl)
it_fs_corr = (funct_conn_it_fs_dms, funct_conn_it_fs_ctl)
it_fr_corr = (funct_conn_it_fr_dms, funct_conn_it_fr_ctl)
it_lit_corr= (funct_conn_it_lit_dms,funct_conn_it_lit_ctl)

rects_v1 = ax.bar(index, it_v1_corr, width, color='purple', label='V1')

rects_v4 = ax.bar(index + width, it_v4_corr, width, color='darkred', label='V4')

rects_fs = ax.bar(index + width*2, it_fs_corr, width, color='lightyellow', label='FS')

rects_d1 = ax.bar(index + width*3, it_d1_corr, width, color='lightblue', label='D1')

rects_d2 = ax.bar(index + width*4, it_d2_corr, width, color='yellow', label='D2')

rects_fr = ax.bar(index + width*5, it_fr_corr, width, color='red', label='FR')

rects_fr = ax.bar(index + width*6, it_lit_corr, width, color='pink', label='LIT')

ax.set_title('FUNCTIONAL CONNECTIVITY OF IT WITH OTHER BRAIN REGIONS (fMRI)')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.set_xlabel('DMS TASK                                        CONTROL TASK')
ax.xaxis.set_label_coords(0.5, -0.025)

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

# Show the plots on the screen
plt.show()
