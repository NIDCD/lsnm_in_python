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
#   Based in part by Matlab scripts by Horwitz et al.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on July 10 2015  
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

from scipy import signal

# define constants needed for hemodynamic function
lambda_ = 6.0

# given the number of total timesteps, calculate total time of scanning
# experiment in seconds
T = 198

# Time for one complete trial in seconds
Ttrial = 5.5

# define neural synaptic time interval and total time of scanning
# experiment (units are seconds)
Ti = .005 * 10

# the scanning happened every Tr interval below (in seconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110
number_of_trials = 36

synaptic_timesteps = experiment_length

# define an array with location of control trials, and another array
# with location of task-related trials, relative to
# an array that contains all trials (task-related trials included)
control_trials = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35]
dms_trials =     [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]

# Load V1 synaptic activity data files into a numpy array
ev1h = np.loadtxt('../visual_model/output/ev1h_synaptic.out')
ev1v = np.loadtxt('../visual_model/output/ev1v_synaptic.out')
iv1h = np.loadtxt('../visual_model/output/iv1h_synaptic.out')
iv1v = np.loadtxt('../visual_model/output/iv1v_synaptic.out')

# Load TVB V1 host node synaptic activity into numpy array
tvb_ev1=np.loadtxt('../visual_model/output/ev1v_tvb_syn.out')
tvb_iv1=np.loadtxt('../visual_model/output/iv1v_tvb_syn.out')

# Load V4 synaptic activity data files into a numpy array
ev4h = np.loadtxt('../visual_model/output/ev4h_synaptic.out')
ev4v = np.loadtxt('../visual_model/output/ev4v_synaptic.out')
ev4c = np.loadtxt('../visual_model/output/ev4c_synaptic.out')
iv4h = np.loadtxt('../visual_model/output/iv4h_synaptic.out')
iv4v = np.loadtxt('../visual_model/output/iv4v_synaptic.out')
iv4c = np.loadtxt('../visual_model/output/iv4c_synaptic.out')

# Load TVB V4 host node synaptic activity into numpy array
tvb_ev4=np.loadtxt('../visual_model/output/ev4v_tvb_syn.out')
tvb_iv4=np.loadtxt('../visual_model/output/iv4v_tvb_syn.out')

# Load IT synaptic activity data files into a numpy array
exss = np.loadtxt('../visual_model/output/exss_synaptic.out')
inss = np.loadtxt('../visual_model/output/inss_synaptic.out')

# Load TVB IT host node synaptic activity into numpy array
tvb_eit=np.loadtxt('../visual_model/output/exss_tvb_syn.out')
tvb_iit=np.loadtxt('../visual_model/output/inss_tvb_syn.out')

# Load D1 synaptic activity data files into a numpy array
efd1 = np.loadtxt('../visual_model/output/efd1_synaptic.out')
ifd1 = np.loadtxt('../visual_model/output/ifd1_synaptic.out')

# Load TVB D1 host node synaptic activity into numpy array
tvb_ed1=np.loadtxt('../visual_model/output/efd1_tvb_syn.out')
tvb_id1=np.loadtxt('../visual_model/output/ifd1_tvb_syn.out')

# Load D2 synaptic activity data files into a numpy array
efd2 = np.loadtxt('../visual_model/output/efd2_synaptic.out')
ifd2 = np.loadtxt('../visual_model/output/ifd2_synaptic.out')

# Load TVB D2 host node synaptic activity into numpy array
tvb_ed2=np.loadtxt('../visual_model/output/efd2_tvb_syn.out')
tvb_id2=np.loadtxt('../visual_model/output/ifd2_tvb_syn.out')

# Load FS synaptic activity data files into a numpy array
exfs = np.loadtxt('../visual_model/output/exfs_synaptic.out')
infs = np.loadtxt('../visual_model/output/infs_synaptic.out')

# Load TVB FS host node synaptic activity into numpy array
tvb_efs=np.loadtxt('../visual_model/output/exfs_tvb_syn.out')
tvb_ifs=np.loadtxt('../visual_model/output/infs_tvb_syn.out')

# Load FR synaptic activity data files into a numpy array
exfr = np.loadtxt('../visual_model/output/exfr_synaptic.out')
infr = np.loadtxt('../visual_model/output/infr_synaptic.out')

# Load TVB FR host node synaptic activity into numpy array
tvb_efr=np.loadtxt('../visual_model/output/exfr_tvb_syn.out')
tvb_ifr=np.loadtxt('../visual_model/output/infr_tvb_syn.out')

# Given neural synaptic time interval and total time of scanning experiment,
# construct a numpy array of time points (data points provided in data files)
time_in_seconds = np.arange(0, T, Tr)

# the following calculates a Poisson distribution (that will represent a hemodynamic
# function, given lambda (the Poisson time constant characterizing width and height
# of hemodynamic function), and tau (the time step)
# if you would do it manually you would do the following:
#h = [lambda_ ** tau * m.exp(-lambda_) / m.factorial(tau) for tau in time_in_seconds]
h = poisson.pmf(time_in_seconds, lambda_)

# resample the array containing the poisson to increase its size and match the size of
# the synaptic activity array
h = signal.resample(h, synaptic_timesteps)

# add all units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
v1 = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1) + tvb_ev1 + tvb_iv1
v4 = np.sum(ev4h + ev4v + ev4c + iv4h + iv4v + iv4c, axis = 1) + tvb_ev4 + tvb_iv4
it = np.sum(exss + inss, axis = 1) + tvb_eit + tvb_iit
d1 = np.sum(efd1 + ifd1, axis = 1) + tvb_ed1 + tvb_ed1
d2 = np.sum(efd2 + ifd2, axis = 1) + tvb_ed2 + tvb_id2
fs = np.sum(exfs + infs, axis = 1) + tvb_efs + tvb_ifs
fr = np.sum(exfr + infr, axis = 1) + tvb_efr + tvb_ifr

# Truncate the final part of time series to match the length of the experiment
# (they are supposed to match anyway, so typically there will be nothing to truncate)
v1 = v1[0:experiment_length]
v4 = v4[0:experiment_length]
it = it[0:experiment_length]
d1 = d1[0:experiment_length]
d2 = d2[0:experiment_length]
fs = fs[0:experiment_length]
fr = fr[0:experiment_length]

# now, we need to convolve the synaptic activity with a hemodynamic delay
# function and sample the array at Tr regular intervals

BOLD_interval = np.arange(0, synaptic_timesteps)

v1_BOLD = np.convolve(v1, h, mode='full')[BOLD_interval]
v4_BOLD = np.convolve(v4, h, mode='full')[BOLD_interval]
it_BOLD = np.convolve(it, h, mode='full')[BOLD_interval]
d1_BOLD = np.convolve(d1, h, mode='full')[BOLD_interval]
d2_BOLD = np.convolve(d2, h, mode='full')[BOLD_interval]
fs_BOLD = np.convolve(fs, h, mode='full')[BOLD_interval]
fr_BOLD = np.convolve(fr, h, mode='full')[BOLD_interval]

# Gets rid of the control trials in the synaptic activity arrays,
# by separating the task-related trials and concatenating them
# together. Remember that each trial is 110 synaptic timesteps
# long.

# first, split the arrays into subarrays, each one containing a single trial
it_subarrays = np.split(it_BOLD, number_of_trials)
v1_subarrays = np.split(v1_BOLD, number_of_trials)
v4_subarrays = np.split(v4_BOLD, number_of_trials)
d1_subarrays = np.split(d1_BOLD, number_of_trials)
d2_subarrays = np.split(d2_BOLD, number_of_trials)
fs_subarrays = np.split(fs_BOLD, number_of_trials)
fr_subarrays = np.split(fr_BOLD, number_of_trials)
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
# ... and concatenate the control task trials together
it_control_trials_ts = np.concatenate(it_control_trials)
v1_control_trials_ts = np.concatenate(v1_control_trials)
v4_control_trials_ts = np.concatenate(v4_control_trials)
d1_control_trials_ts = np.concatenate(d1_control_trials)
d2_control_trials_ts = np.concatenate(d2_control_trials)
fs_control_trials_ts = np.concatenate(fs_control_trials)
fr_control_trials_ts = np.concatenate(fr_control_trials)

# Downsample the resulting fMRI timeseries for both DMS and control trials
# Convert seconds to Ti units (how many times the scanning interval fits into each
# synaptic interval)

Tr_new = round(Tr / Ti)

# We need to rescale the BOLD signal arrays to match the timescale of the synaptic
# signals. We also truncate the resulting float down to the nearest integer. in other
# words, we are downsampling the BOLD array to match the scan interval time Tr

BOLD_timing = m.trunc(v1_DMS_trials_ts.size / Tr_new)

v1_DMS_trials_ts = [v1_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
v4_DMS_trials_ts = [v4_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
it_DMS_trials_ts = [it_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
d1_DMS_trials_ts = [d1_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
d2_DMS_trials_ts = [d2_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
fs_DMS_trials_ts = [fs_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
fr_DMS_trials_ts = [fr_DMS_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
v1_control_trials_ts = [v1_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
v4_control_trials_ts = [v4_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
it_control_trials_ts = [it_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
d1_control_trials_ts = [d1_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
d2_control_trials_ts = [d2_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
fs_control_trials_ts = [fs_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]
fr_control_trials_ts = [fr_control_trials_ts[i * Tr_new + 1] for i in np.arange(BOLD_timing)]

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


# now, group the values to be plotted by brain module
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