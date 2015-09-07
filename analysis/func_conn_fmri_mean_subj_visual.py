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
#   This file (func_conn_fmri_mean_subj_visual.py) was created September 6, 2015.
#
#   Based in part on Matlab scripts by Horwitz et al.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 6, 2015  
# **************************************************************************/

# func_conn_fmri_mean_subj_visual.py
#
# Calculate and plot functional connectivity (within-task time series correlation)
# of the mean BOLD (and synaptic) timeseries in IT with the BOLD (and synaptic) timeseries
# of all other simulated brain
# regions. It reads the BOLD (and synaptic) time-series from a  single python data file (*.npy)
# and writes the correlation coefficients to an output data file (*.npy)
#
# For the visual model, the columns in input files are in the following order:
#
# V1, V4, IT, FS, D1, D2, FR

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import math as m

from scipy.stats import poisson

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110

num_of_trials = 36             # number of trials

num_of_fmri_blocks = 11             # how many blocks of trials in the experiment
num_of_syn_blocks = 12              # fMRI blocks are less than synaptic blocks
                                    # because blocks are removed from fmri time-series

scans_removed = 8             # number of scans removed from BOLD computation

num_of_modules = 7

synaptic_timesteps = experiment_length

# define the names of the input file where the synaptic and BOLD timeseries are contained
syn_file  = 'avg_syn_across_subjs.npy'
BOLD_file = 'avg_BOLD_across_subjs_balloon.npy'

# define the name of the output file where the functional connectivity
# correlation coefficients will be stored
func_conn_dms_file = 'corr_fmri_IT_vs_all_dms.npy'
func_conn_ctl_file = 'corr_fmri_IT_vs_all_ctl.npy'

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
synaptic_timescale = np.arange(0, synaptic_timesteps)

# open files containing synaptic activities and BOLD time-series
syn_ts  = np.load(syn_file)
BOLD_ts = np.load(BOLD_file)    

# now, split all time-series in blocks
syn_ts_v1=np.array_split(syn_ts[0], num_of_syn_blocks)
syn_ts_v4=np.array_split(syn_ts[1], num_of_syn_blocks)
syn_ts_it=np.array_split(syn_ts[2], num_of_syn_blocks)
syn_ts_fs=np.array_split(syn_ts[3], num_of_syn_blocks)
syn_ts_d1=np.array_split(syn_ts[4], num_of_syn_blocks)
syn_ts_d2=np.array_split(syn_ts[5], num_of_syn_blocks)
syn_ts_fr=np.array_split(syn_ts[6], num_of_syn_blocks)
BOLD_ts_v1=np.array_split(BOLD_ts[0], num_of_fmri_blocks)
BOLD_ts_v4=np.array_split(BOLD_ts[1], num_of_fmri_blocks)
BOLD_ts_it=np.array_split(BOLD_ts[2], num_of_fmri_blocks)
BOLD_ts_fs=np.array_split(BOLD_ts[3], num_of_fmri_blocks)
BOLD_ts_d1=np.array_split(BOLD_ts[4], num_of_fmri_blocks)
BOLD_ts_d2=np.array_split(BOLD_ts[5], num_of_fmri_blocks)
BOLD_ts_fr=np.array_split(BOLD_ts[6], num_of_fmri_blocks)

# define an array with location of control blocks, and another array
# with location of task (DMS) blocks, relative to
# an array that contains all blocks (task-related blocks included)
syn_control_block_ids = np.arange(1, num_of_syn_blocks, 2)
syn_dms_block_ids =     np.arange(0, num_of_syn_blocks, 2)
fmri_control_block_ids = np.arange(0, num_of_fmri_blocks, 2)
fmri_dms_block_ids =     np.arange(1, num_of_fmri_blocks, 2)

# now, create an array of synaptic time-series containing DMS trials only: 
syn_v1_dms_blocks = np.delete(np.asarray(syn_ts_v1), syn_control_block_ids, axis=0)
syn_v4_dms_blocks = np.delete(np.asarray(syn_ts_v4), syn_control_block_ids, axis=0)
syn_it_dms_blocks = np.delete(np.asarray(syn_ts_it), syn_control_block_ids, axis=0)
syn_fs_dms_blocks = np.delete(np.asarray(syn_ts_fs), syn_control_block_ids, axis=0)
syn_d1_dms_blocks = np.delete(np.asarray(syn_ts_d1), syn_control_block_ids, axis=0)
syn_d2_dms_blocks = np.delete(np.asarray(syn_ts_d2), syn_control_block_ids, axis=0)
syn_fr_dms_blocks = np.delete(np.asarray(syn_ts_fr), syn_control_block_ids, axis=0)
# now, create an array of BOLD time-series containing DMS trials only: 
BOLD_v1_dms_blocks = np.delete(np.asarray(BOLD_ts_v1), fmri_control_block_ids, axis=0)
BOLD_v4_dms_blocks = np.delete(np.asarray(BOLD_ts_v4), fmri_control_block_ids, axis=0)
BOLD_it_dms_blocks = np.delete(np.asarray(BOLD_ts_it), fmri_control_block_ids, axis=0)
BOLD_fs_dms_blocks = np.delete(np.asarray(BOLD_ts_fs), fmri_control_block_ids, axis=0)
BOLD_d1_dms_blocks = np.delete(np.asarray(BOLD_ts_d1), fmri_control_block_ids, axis=0)
BOLD_d2_dms_blocks = np.delete(np.asarray(BOLD_ts_d2), fmri_control_block_ids, axis=0)
BOLD_fr_dms_blocks = np.delete(np.asarray(BOLD_ts_fr), fmri_control_block_ids, axis=0)

# ... and concatenate those DMS timeseries together
syn_v1_dms_ts = np.concatenate(syn_v1_dms_blocks)
syn_v4_dms_ts = np.concatenate(syn_v4_dms_blocks)
syn_it_dms_ts = np.concatenate(syn_it_dms_blocks)
syn_fs_dms_ts = np.concatenate(syn_fs_dms_blocks)
syn_d1_dms_ts = np.concatenate(syn_d1_dms_blocks)
syn_d2_dms_ts = np.concatenate(syn_d2_dms_blocks)
syn_fr_dms_ts = np.concatenate(syn_fr_dms_blocks)

BOLD_v1_dms_ts = np.concatenate(BOLD_v1_dms_blocks)
BOLD_v4_dms_ts = np.concatenate(BOLD_v4_dms_blocks)
BOLD_it_dms_ts = np.concatenate(BOLD_it_dms_blocks)
BOLD_fs_dms_ts = np.concatenate(BOLD_fs_dms_blocks)
BOLD_d1_dms_ts = np.concatenate(BOLD_d1_dms_blocks)
BOLD_d2_dms_ts = np.concatenate(BOLD_d2_dms_blocks)
BOLD_fr_dms_ts = np.concatenate(BOLD_fr_dms_blocks)

# but also, get rid of the DMS blocks, to create arrays with only control trials
syn_v1_control_blocks = np.delete(np.asarray(syn_ts_v1), syn_dms_block_ids, axis=0)
syn_v4_control_blocks = np.delete(np.asarray(syn_ts_v4), syn_dms_block_ids, axis=0)
syn_it_control_blocks = np.delete(np.asarray(syn_ts_it), syn_dms_block_ids, axis=0)
syn_fs_control_blocks = np.delete(np.asarray(syn_ts_fs), syn_dms_block_ids, axis=0)
syn_d1_control_blocks = np.delete(np.asarray(syn_ts_d1), syn_dms_block_ids, axis=0)
syn_d2_control_blocks = np.delete(np.asarray(syn_ts_d2), syn_dms_block_ids, axis=0)
syn_fr_control_blocks = np.delete(np.asarray(syn_ts_fr), syn_dms_block_ids, axis=0)

BOLD_v1_control_blocks = np.delete(np.asarray(BOLD_ts_v1), fmri_dms_block_ids, axis=0)
BOLD_v4_control_blocks = np.delete(np.asarray(BOLD_ts_v4), fmri_dms_block_ids, axis=0)
BOLD_it_control_blocks = np.delete(np.asarray(BOLD_ts_it), fmri_dms_block_ids, axis=0)
BOLD_fs_control_blocks = np.delete(np.asarray(BOLD_ts_fs), fmri_dms_block_ids, axis=0)
BOLD_d1_control_blocks = np.delete(np.asarray(BOLD_ts_d1), fmri_dms_block_ids, axis=0)
BOLD_d2_control_blocks = np.delete(np.asarray(BOLD_ts_d2), fmri_dms_block_ids, axis=0)
BOLD_fr_control_blocks = np.delete(np.asarray(BOLD_ts_fr), fmri_dms_block_ids, axis=0)

# ... and concatenate the control blocks together
syn_v1_ctl_ts = np.concatenate(syn_v1_control_blocks)
syn_v4_ctl_ts = np.concatenate(syn_v4_control_blocks)
syn_it_ctl_ts = np.concatenate(syn_it_control_blocks)
syn_fs_ctl_ts = np.concatenate(syn_fs_control_blocks)
syn_d1_ctl_ts = np.concatenate(syn_d1_control_blocks)
syn_d2_ctl_ts = np.concatenate(syn_d2_control_blocks)
syn_fr_ctl_ts = np.concatenate(syn_fr_control_blocks)

BOLD_v1_ctl_ts = np.concatenate(BOLD_v1_control_blocks)
BOLD_v4_ctl_ts = np.concatenate(BOLD_v4_control_blocks)
BOLD_it_ctl_ts = np.concatenate(BOLD_it_control_blocks)
BOLD_fs_ctl_ts = np.concatenate(BOLD_fs_control_blocks)
BOLD_d1_ctl_ts = np.concatenate(BOLD_d1_control_blocks)
BOLD_d2_ctl_ts = np.concatenate(BOLD_d2_control_blocks)
BOLD_fr_ctl_ts = np.concatenate(BOLD_fr_control_blocks)

# now, convert DMS and control timeseries into pandas timeseries, so we can analyze it
syn_V1_dms_ts = pd.Series(syn_v1_dms_ts)
syn_V4_dms_ts = pd.Series(syn_v4_dms_ts)
syn_IT_dms_ts = pd.Series(syn_it_dms_ts)
syn_FS_dms_ts = pd.Series(syn_fs_dms_ts)
syn_D1_dms_ts = pd.Series(syn_d1_dms_ts)
syn_D2_dms_ts = pd.Series(syn_d2_dms_ts)
syn_FR_dms_ts = pd.Series(syn_fr_dms_ts)

BOLD_V1_dms_ts = pd.Series(BOLD_v1_dms_ts)
BOLD_V4_dms_ts = pd.Series(BOLD_v4_dms_ts)
BOLD_IT_dms_ts = pd.Series(BOLD_it_dms_ts)
BOLD_FS_dms_ts = pd.Series(BOLD_fs_dms_ts)
BOLD_D1_dms_ts = pd.Series(BOLD_d1_dms_ts)
BOLD_D2_dms_ts = pd.Series(BOLD_d2_dms_ts)
BOLD_FR_dms_ts = pd.Series(BOLD_fr_dms_ts)

syn_V1_ctl_ts = pd.Series(syn_v1_ctl_ts)
syn_V4_ctl_ts = pd.Series(syn_v4_ctl_ts)
syn_IT_ctl_ts = pd.Series(syn_it_ctl_ts)
syn_FS_ctl_ts = pd.Series(syn_fs_ctl_ts)
syn_D1_ctl_ts = pd.Series(syn_d1_ctl_ts)
syn_D2_ctl_ts = pd.Series(syn_d2_ctl_ts)
syn_FR_ctl_ts = pd.Series(syn_fr_ctl_ts)

BOLD_V1_ctl_ts = pd.Series(BOLD_v1_ctl_ts)
BOLD_V4_ctl_ts = pd.Series(BOLD_v4_ctl_ts)
BOLD_IT_ctl_ts = pd.Series(BOLD_it_ctl_ts)
BOLD_FS_ctl_ts = pd.Series(BOLD_fs_ctl_ts)
BOLD_D1_ctl_ts = pd.Series(BOLD_d1_ctl_ts)
BOLD_D2_ctl_ts = pd.Series(BOLD_d2_ctl_ts)
BOLD_FR_ctl_ts = pd.Series(BOLD_fr_ctl_ts)

# ... and calculate the functional connectivity of IT with the other modules
funct_conn_it_v1_dms_SYN = syn_IT_dms_ts.corr(syn_V1_dms_ts)
funct_conn_it_v4_dms_SYN = syn_IT_dms_ts.corr(syn_V4_dms_ts)
funct_conn_it_d1_dms_SYN = syn_IT_dms_ts.corr(syn_D1_dms_ts)
funct_conn_it_d2_dms_SYN = syn_IT_dms_ts.corr(syn_D2_dms_ts)
funct_conn_it_fs_dms_SYN = syn_IT_dms_ts.corr(syn_FS_dms_ts)
funct_conn_it_fr_dms_SYN = syn_IT_dms_ts.corr(syn_FR_dms_ts)

funct_conn_it_v1_dms_BOLD = BOLD_IT_dms_ts.corr(BOLD_V1_dms_ts)
funct_conn_it_v4_dms_BOLD = BOLD_IT_dms_ts.corr(BOLD_V4_dms_ts)
funct_conn_it_d1_dms_BOLD = BOLD_IT_dms_ts.corr(BOLD_D1_dms_ts)
funct_conn_it_d2_dms_BOLD = BOLD_IT_dms_ts.corr(BOLD_D2_dms_ts)
funct_conn_it_fs_dms_BOLD = BOLD_IT_dms_ts.corr(BOLD_FS_dms_ts)
funct_conn_it_fr_dms_BOLD = BOLD_IT_dms_ts.corr(BOLD_FR_dms_ts)

funct_conn_it_v1_ctl_SYN = syn_IT_ctl_ts.corr(syn_V1_ctl_ts)
funct_conn_it_v4_ctl_SYN = syn_IT_ctl_ts.corr(syn_V4_ctl_ts)
funct_conn_it_d1_ctl_SYN = syn_IT_ctl_ts.corr(syn_D1_ctl_ts)
funct_conn_it_d2_ctl_SYN = syn_IT_ctl_ts.corr(syn_D2_ctl_ts)
funct_conn_it_fs_ctl_SYN = syn_IT_ctl_ts.corr(syn_FS_ctl_ts)
funct_conn_it_fr_ctl_SYN = syn_IT_ctl_ts.corr(syn_FR_ctl_ts)

funct_conn_it_v1_ctl_BOLD = BOLD_IT_ctl_ts.corr(BOLD_V1_ctl_ts)
funct_conn_it_v4_ctl_BOLD = BOLD_IT_ctl_ts.corr(BOLD_V4_ctl_ts)
funct_conn_it_d1_ctl_BOLD = BOLD_IT_ctl_ts.corr(BOLD_D1_ctl_ts)
funct_conn_it_d2_ctl_BOLD = BOLD_IT_ctl_ts.corr(BOLD_D2_ctl_ts)
funct_conn_it_fs_ctl_BOLD = BOLD_IT_ctl_ts.corr(BOLD_FS_ctl_ts)
funct_conn_it_fr_ctl_BOLD = BOLD_IT_ctl_ts.corr(BOLD_FR_ctl_ts)

#func_conn_dms_syn  = np.array([funct_conn_it_v1_dms_SYN,funct_conn_it_v4_dms_SYN,
#                               funct_conn_it_d1_dms_SYN,funct_conn_it_d2_dms_SYN,
#                               funct_conn_it_fs_dms_SYN,funct_conn_it_fr_dms_SYN])
#func_conn_ctl_syn  = np.array([funct_conn_it_v1_ctl_SYN,funct_conn_it_v4_ctl_SYN,
#                               funct_conn_it_d1_ctl_SYN,funct_conn_it_d2_ctl_SYN,
#                               funct_conn_it_fs_ctl_SYN,funct_conn_it_fr_ctl_SYN])
#
#func_conn_dms_fmri = np.array([funct_conn_it_v1_dms_BOLD,funct_conn_it_v4_dms_BOLD,
#                               funct_conn_it_d1_dms_BOLD,funct_conn_it_d2_dms_BOLD,
#                               funct_conn_it_fs_dms_BOLD,funct_conn_it_fr_dms_BOLD])
#func_conn_ctl_fmri = np.array([funct_conn_it_v1_ctl_BOLD,funct_conn_it_v4_ctl_BOLD,
#                               funct_conn_it_d1_ctl_BOLD,funct_conn_it_d2_ctl_BOLD,
#                               funct_conn_it_fs_ctl_BOLD,funct_conn_it_fr_ctl_BOLD])
#
# now, save all correlation coefficients to a output files 
#np.save(func_conn_dms_file, func_conn_dms)
#np.save(func_conn_ctl_file, func_conn_ctl)

# now, group the values to be plotted by brain module
it_v1_corr_syn = (funct_conn_it_v1_dms_SYN, funct_conn_it_v1_ctl_SYN)
it_v4_corr_syn = (funct_conn_it_v4_dms_SYN, funct_conn_it_v4_ctl_SYN)
it_d1_corr_syn = (funct_conn_it_d1_dms_SYN, funct_conn_it_d1_ctl_SYN)
it_d2_corr_syn = (funct_conn_it_d2_dms_SYN, funct_conn_it_d2_ctl_SYN)
it_fs_corr_syn = (funct_conn_it_fs_dms_SYN, funct_conn_it_fs_ctl_SYN)
it_fr_corr_syn = (funct_conn_it_fr_dms_SYN, funct_conn_it_fr_ctl_SYN)

it_v1_corr_fmri = (funct_conn_it_v1_dms_BOLD, funct_conn_it_v1_ctl_BOLD)
it_v4_corr_fmri = (funct_conn_it_v4_dms_BOLD, funct_conn_it_v4_ctl_BOLD)
it_d1_corr_fmri = (funct_conn_it_d1_dms_BOLD, funct_conn_it_d1_ctl_BOLD)
it_d2_corr_fmri = (funct_conn_it_d2_dms_BOLD, funct_conn_it_d2_ctl_BOLD)
it_fs_corr_fmri = (funct_conn_it_fs_dms_BOLD, funct_conn_it_fs_ctl_BOLD)
it_fr_corr_fmri = (funct_conn_it_fr_dms_BOLD, funct_conn_it_fr_ctl_BOLD)

# define number of groups to plot
N = 2

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([0,1])

rects_v1 = ax.bar(index, it_v1_corr_syn, width, color='purple', label='V1')

rects_v4 = ax.bar(index + width, it_v4_corr_syn, width, color='darkred', label='V4')

rects_fs = ax.bar(index + width*2, it_fs_corr_syn, width, color='lightyellow', label='FS')

rects_d1 = ax.bar(index + width*3, it_d1_corr_syn, width, color='lightblue', label='D1')

rects_d2 = ax.bar(index + width*4, it_d2_corr_syn, width, color='yellow', label='D2')

rects_fr = ax.bar(index + width*5, it_fr_corr_syn, width, color='red', label='FR')

ax.set_title('FUNCTIONAL CONNECTIVITY OF IT WITH OTHER BRAIN REGIONS (SYNAPTIC)')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.set_xlabel('DMS TASK                                        CONTROL TASK')
ax.xaxis.set_label_coords(0.5, -0.025)

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

# create a new figure
fig, ax = plt.subplots()

ax.set_ylim([0,1])


rects_v1 = ax.bar(index, it_v1_corr_fmri, width, color='purple', label='V1')

rects_v4 = ax.bar(index + width, it_v4_corr_fmri, width, color='darkred', label='V4')

rects_fs = ax.bar(index + width*2, it_fs_corr_fmri, width, color='lightyellow', label='FS')

rects_d1 = ax.bar(index + width*3, it_d1_corr_fmri, width, color='lightblue', label='D1')

rects_d2 = ax.bar(index + width*4, it_d2_corr_fmri, width, color='yellow', label='D2')

rects_fr = ax.bar(index + width*5, it_fr_corr_fmri, width, color='red', label='FR')

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
