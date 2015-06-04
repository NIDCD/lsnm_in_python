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
#   This file (functionalConnectivityVisual.py) was created on May 4, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on May 4 2015  
# **************************************************************************/

# functionalConnectivityVisual.py
#
# Calculate and plot functional connectivity (within-task time series correlation)
# of IT with all other simulated brain areas, using the output 
# from visual DMS task (synaptic activity and BOLD activity time series)

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 5280
trial_length = 440
number_of_trials = 12

# define an array with location of control trials, and another array
# with location of task-related trials, relative to
# an array that contains all trials (task-related trials included)
control_trials = [1, 3, 5, 7, 9, 11]
dms_trials =     [0, 2, 4, 6, 8, 10]

# Load V1 synaptic activity data files into a numpy array
ev1h = np.loadtxt('../../output/ev1h_synaptic.out')
ev1v = np.loadtxt('../../output/ev1v_synaptic.out')
iv1h = np.loadtxt('../../output/iv1h_synaptic.out')
iv1v = np.loadtxt('../../output/iv1v_synaptic.out')

# Load V4 synaptic activity data files into a numpy array
ev4h = np.loadtxt('../../output/ev4h_synaptic.out')
ev4v = np.loadtxt('../../output/ev4v_synaptic.out')
ev4c = np.loadtxt('../../output/ev4c_synaptic.out')
iv4h = np.loadtxt('../../output/iv4h_synaptic.out')
iv4v = np.loadtxt('../../output/iv4v_synaptic.out')
iv4c = np.loadtxt('../../output/iv4c_synaptic.out')

# Load IT synaptic activity data files into a numpy array
exss = np.loadtxt('../../output/exss_synaptic.out')
inss = np.loadtxt('../../output/inss_synaptic.out')

# Load D1 synaptic activity data files into a numpy array
efd1 = np.loadtxt('../../output/efd1_synaptic.out')
ifd1 = np.loadtxt('../../output/ifd1_synaptic.out')

# Load D2 synaptic activity data files into a numpy array
efd2 = np.loadtxt('../../output/efd2_synaptic.out')
ifd2 = np.loadtxt('../../output/ifd2_synaptic.out')

# Load FS synaptic activity data files into a numpy array
exfs = np.loadtxt('../../output/exfs_synaptic.out')
infs = np.loadtxt('../../output/infs_synaptic.out')

# Load FR synaptic activity data files into a numpy array
exfr = np.loadtxt('../../output/exfr_synaptic.out')
infr = np.loadtxt('../../output/infr_synaptic.out')

# add all units within each region (V1, IT, and D1) together across space to calculate
# synaptic activity in each brain region
v1 = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1)
v4 = np.sum(ev4h + ev4v + ev4c + iv4h + iv4v + iv4c, axis = 1)
it = np.sum(exss + inss, axis = 1)
d1 = np.sum(efd1 + ifd1, axis = 1)
d2 = np.sum(efd2 + ifd2, axis = 1)
fs = np.sum(exfs + infs, axis = 1)
fr = np.sum(exfr + infr, axis = 1)

# Gets rid of the control trials in the synaptic activity arrays,
# by separating the task-related trials and concatenating them
# together. Remember that each trial is 440 synaptic timesteps
# long.

# first, split the arrays into subarrays, each one containing a single trial
it_subarrays = np.split(it, number_of_trials)
v1_subarrays = np.split(v1, number_of_trials)
v4_subarrays = np.split(v4, number_of_trials)
d1_subarrays = np.split(d1, number_of_trials)
d2_subarrays = np.split(d2, number_of_trials)
fs_subarrays = np.split(fs, number_of_trials)
fr_subarrays = np.split(fr, number_of_trials)
# now, get rid of the control trials...
it_DMS_trials = np.delete(it_subarrays, control_trials, axis=0)
v1_DMS_trials = np.delete(v1_subarrays, control_trials, axis=0)
v4_DMS_trials = np.delete(v4_subarrays, control_trials, axis=0)
d1_DMS_trials = np.delete(d1_subarrays, control_trials, axis=0)
d2_DMS_trials = np.delete(d2_subarrays, control_trials, axis=0)
fs_DMS_trials = np.delete(fs_subarrays, control_trials, axis=0)
fr_DMS_trials = np.delete(fr_subarrays, control_trials, axis=0)
# ... and concatenate the task-related trials together
it_DMS_trials_ts = np.concatenate(it_DMS_trials)
v1_DMS_trials_ts = np.concatenate(v1_DMS_trials)
v4_DMS_trials_ts = np.concatenate(v4_DMS_trials)
d1_DMS_trials_ts = np.concatenate(d1_DMS_trials)
d2_DMS_trials_ts = np.concatenate(d2_DMS_trials)
fs_DMS_trials_ts = np.concatenate(fs_DMS_trials)
fr_DMS_trials_ts = np.concatenate(fr_DMS_trials)

# but also, get rid of the DMS task trials, to create arrays with only control trials
it_control_trials = np.delete(it_subarrays, dms_trials, axis=0)
v1_control_trials = np.delete(v1_subarrays, dms_trials, axis=0)
v4_control_trials = np.delete(v4_subarrays, dms_trials, axis=0)
d1_control_trials = np.delete(d1_subarrays, dms_trials, axis=0)
d2_control_trials = np.delete(d2_subarrays, dms_trials, axis=0)
fs_control_trials = np.delete(fs_subarrays, dms_trials, axis=0)
fr_control_trials = np.delete(fr_subarrays, dms_trials, axis=0)
# ... and contatenate the control task trials together
it_control_trials_ts = np.concatenate(it_control_trials)
v1_control_trials_ts = np.concatenate(v1_control_trials)
v4_control_trials_ts = np.concatenate(v4_control_trials)
d1_control_trials_ts = np.concatenate(d1_control_trials)
d2_control_trials_ts = np.concatenate(d2_control_trials)
fs_control_trials_ts = np.concatenate(fs_control_trials)
fr_control_trials_ts = np.concatenate(fr_control_trials)


# now, convert DMS and control timeseries into pandas timeseries, so we can analyze it
IT_dms_ts = pd.Series(it_DMS_trials_ts)
V1_dms_ts = pd.Series(v1_DMS_trials_ts)
V4_dms_ts = pd.Series(v4_DMS_trials_ts)
D1_dms_ts = pd.Series(d1_DMS_trials_ts)
D2_dms_ts = pd.Series(d2_DMS_trials_ts)
FS_dms_ts = pd.Series(fs_DMS_trials_ts)
FR_dms_ts = pd.Series(fr_DMS_trials_ts)

IT_ctl_ts = pd.Series(it_control_trials_ts)
V1_ctl_ts = pd.Series(v1_control_trials_ts)
V4_ctl_ts = pd.Series(v4_control_trials_ts)
D1_ctl_ts = pd.Series(d1_control_trials_ts)
D2_ctl_ts = pd.Series(d2_control_trials_ts)
FS_ctl_ts = pd.Series(fs_control_trials_ts)
FR_ctl_ts = pd.Series(fr_control_trials_ts)


# ... and calculate the functional connectivity of IT with the other modules
funct_conn_it_v1_dms = IT_dms_ts.corr(V1_dms_ts)
funct_conn_it_v4_dms = IT_dms_ts.corr(V4_dms_ts)
funct_conn_it_d1_dms = IT_dms_ts.corr(D1_dms_ts)
funct_conn_it_d2_dms = IT_dms_ts.corr(D2_dms_ts)
funct_conn_it_fs_dms = IT_dms_ts.corr(FS_dms_ts)
funct_conn_it_fr_dms = IT_dms_ts.corr(FR_dms_ts)

funct_conn_it_v1_ctl = IT_ctl_ts.corr(V1_ctl_ts)
funct_conn_it_v4_ctl = IT_ctl_ts.corr(V4_ctl_ts)
funct_conn_it_d1_ctl = IT_ctl_ts.corr(D1_ctl_ts)
funct_conn_it_d2_ctl = IT_ctl_ts.corr(D2_ctl_ts)
funct_conn_it_fs_ctl = IT_ctl_ts.corr(FS_ctl_ts)
funct_conn_it_fr_ctl = IT_ctl_ts.corr(FR_ctl_ts)


# define number of groups to plot
N = 2

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()


# now, group the vaules to be plotted by brain module
it_v1_corr = (funct_conn_it_v1_dms, funct_conn_it_v1_ctl)
it_v4_corr = (funct_conn_it_v4_dms, funct_conn_it_v4_ctl)
it_d1_corr = (funct_conn_it_d1_dms, funct_conn_it_d1_ctl)
it_d2_corr = (funct_conn_it_d2_dms, funct_conn_it_d2_ctl)
it_fs_corr = (funct_conn_it_fs_dms, funct_conn_it_fs_ctl)
it_fr_corr = (funct_conn_it_fr_dms, funct_conn_it_fr_ctl)

rects_v1 = ax.bar(index, it_v1_corr, width, color='purple', label='V1')

rects_v4 = ax.bar(index + width, it_v4_corr, width, color='darkred', label='V4')

rects_fs = ax.bar(index + width*2, it_fs_corr, width, color='lightyellow', label='FS')

rects_d1 = ax.bar(index + width*3, it_d1_corr, width, color='lightblue', label='D1')

rects_d2 = ax.bar(index + width*4, it_d2_corr, width, color='yellow', label='D2')

rects_fr = ax.bar(index + width*5, it_fr_corr, width, color='red', label='FR')

ax.set_title('FUNCTIONAL CONNECTIVITY OF IT WITH ALL OTHER BRAIN REGIONS')

# Show the plot on the screen
plt.show()
