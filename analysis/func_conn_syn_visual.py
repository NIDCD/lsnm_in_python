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
#   This file (func_conn_syn_visual.py) was created on May 4, 2015.
#
#   Based in part by Matlab scripts by Horwitz et al.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 7 2015  
# **************************************************************************/

# func_conn_syn_visual.py
#
# Calculate and plot functional connectivity (within-task time series correlation)
# of IT with all other simulated brain areas, using the output 
# from visual DMS task (integrated synaptic activity) on a single subject.

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

# set matplotlib parameters to produce visually appealing plots
mpl.style.use('ggplot')

#def plot_corr(df):
#    '''Plots correlation matrix, taking a pandas DataFrame as input
#    '''
#    
#    corr = df.corr()
#    plt.matshow(corr)
#    plt.xticks(range(len(corr.columns)), corr.columns);
#    plt.yticks(range(len(corr.columns)), corr.columns);
#    plt.colorbar()

# what are the locations of relevant TVB nodes within TVB array?
#v1_loc = 345
#v4_loc = 393
#it_loc = 413
#pf_loc =  74

# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_ROI_test_ignore.npy'

# define the name of the output file where the functional connectivity timeseries will be stored
func_conn_dms_file = 'corr_syn_IT_vs_all_dms_test_ignore.npy'
func_conn_ctl_file = 'corr_syn_IT_vs_all_ctl_test_ignore.npy'

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110
number_of_trials = 36

# define intertrial interval duration in number of synaptic timesteps
ITI_length = 20

# define an array with location of control trials, and another array
# with location of task-related trials, relative to
# an array that contains all trials (task-related trials included)
# We will use this array to split the synaptic activity timeseries
# into separate trials.
control_trials = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]
dms_trials =     [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32]

# open file that contains the synaptic activities
syn = np.load(SYN_file)

# extract synaptic activities for each ROI
v1_syn = syn[0]
v4_syn = syn[1]
it_syn = syn[2]
fs_syn = syn[3]
d1_syn = syn[4]
d2_syn = syn[5]
fr_syn = syn[6]
lit_syn= syn[7]

# Gets rid of the control trials in the synaptic activity arrays,
# by separating the task-related trials and concatenating them
# together. Remember that each trial is a number of synaptic timesteps
# long.

# first, split the arrays into subarrays, each one containing a single trial
it_subarrays = np.split(it_syn, number_of_trials)
v1_subarrays = np.split(v1_syn, number_of_trials)
v4_subarrays = np.split(v4_syn, number_of_trials)
d1_subarrays = np.split(d1_syn, number_of_trials)
d2_subarrays = np.split(d2_syn, number_of_trials)
fs_subarrays = np.split(fs_syn, number_of_trials)
fr_subarrays = np.split(fr_syn, number_of_trials)
lit_subarrays= np.split(lit_syn,number_of_trials)

# we get rid of the inter-trial interval for each and all trials
# 1 second at the end of each trial.
# 1 second = 20 synaptic timesteps
#ITI_start = trial_length - ITI_length
# Get rid of the ITI, located at the end of each trial
#it_subarrays = np.delete(it_subarrays, np.arange(ITI_start,trial_length), axis=1)
#v1_subarrays = np.delete(v1_subarrays, np.arange(ITI_start,trial_length), axis=1)
#v4_subarrays = np.delete(v4_subarrays, np.arange(ITI_start,trial_length), axis=1)
#d1_subarrays = np.delete(d1_subarrays, np.arange(ITI_start,trial_length), axis=1)
#d2_subarrays = np.delete(d2_subarrays, np.arange(ITI_start,trial_length), axis=1)
#fs_subarrays = np.delete(fs_subarrays, np.arange(ITI_start,trial_length), axis=1)
#fr_subarrays = np.delete(fr_subarrays, np.arange(ITI_start,trial_length), axis=1)

#it_subarrays = np.delete(it_subarrays, np.arange(0,ITI_length), axis=1)
#v1_subarrays = np.delete(v1_subarrays, np.arange(0,ITI_length), axis=1)
#v4_subarrays = np.delete(v4_subarrays, np.arange(0,ITI_length), axis=1)
#d1_subarrays = np.delete(d1_subarrays, np.arange(0,ITI_length), axis=1)
#d2_subarrays = np.delete(d2_subarrays, np.arange(0,ITI_length), axis=1)
#fs_subarrays = np.delete(fs_subarrays, np.arange(0,ITI_length), axis=1)
#fr_subarrays = np.delete(fr_subarrays, np.arange(0,ITI_length), axis=1)

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
it_DMS_trials_ts = np.concatenate(it_DMS_trials)
v1_DMS_trials_ts = np.concatenate(v1_DMS_trials)
v4_DMS_trials_ts = np.concatenate(v4_DMS_trials)
d1_DMS_trials_ts = np.concatenate(d1_DMS_trials)
d2_DMS_trials_ts = np.concatenate(d2_DMS_trials)
fs_DMS_trials_ts = np.concatenate(fs_DMS_trials)
fr_DMS_trials_ts = np.concatenate(fr_DMS_trials)
lit_DMS_trials_ts= np.concatenate(lit_DMS_trials)

# but also, get rid of the DMS task trials, to create arrays that contain only control trials
it_control_trials = np.delete(it_subarrays, dms_trials, axis=0)
v1_control_trials = np.delete(v1_subarrays, dms_trials, axis=0)
v4_control_trials = np.delete(v4_subarrays, dms_trials, axis=0)
d1_control_trials = np.delete(d1_subarrays, dms_trials, axis=0)
d2_control_trials = np.delete(d2_subarrays, dms_trials, axis=0)
fs_control_trials = np.delete(fs_subarrays, dms_trials, axis=0)
fr_control_trials = np.delete(fr_subarrays, dms_trials, axis=0)
lit_control_trials= np.delete(lit_subarrays,dms_trials, axis=0)

# ... and concatenate the control task trials together
it_control_trials_ts = np.concatenate(it_control_trials)
v1_control_trials_ts = np.concatenate(v1_control_trials)
v4_control_trials_ts = np.concatenate(v4_control_trials)
d1_control_trials_ts = np.concatenate(d1_control_trials)
d2_control_trials_ts = np.concatenate(d2_control_trials)
fs_control_trials_ts = np.concatenate(fs_control_trials)
fr_control_trials_ts = np.concatenate(fr_control_trials)
lit_control_trials_ts= np.concatenate(lit_control_trials)

############ TMP BEGINS
# put all of the time-series together in preparation forthe correlation analysis
#dms_trials = np.array([it_DMS_trials_ts, v1_DMS_trials_ts, v4_DMS_trials_ts,
#                       d1_DMS_trials_ts, d2_DMS_trials_ts, fs_DMS_trials_ts,
#                       fr_DMS_trials_ts])
#ctl_trials = np.array([it_control_trials_ts, v1_control_trials_ts, v4_control_trials_ts,
#                       d1_control_trials_ts, d2_control_trials_ts, fs_control_trials_ts,
#                       fr_control_trials_ts])
# extract the length of the time-series
#ts_length = it_DMS_trials_ts.size
# convert to Pandas dataframe, using the transpose to convert to a format where the names
# of the modules are the labels for each time-series
#dms_ts = pd.DataFrame(dms_trials.T,
#                      columns=np.array(['V1', 'V4', 'IT', 'D1', 'D2', 'FS', 'FR']),
#                      index=list(range(ts_length)) )
#ctl_ts = pd.DataFrame(ctl_trials.T,
#                      columns=np.array(['V1', 'V4', 'IT', 'D1', 'D2', 'FS', 'FR']),
#                      index=list(range(ts_length)) )
#plot_corr(dms_ts)
#plot_corr(ctl_ts)
#plt.show()
############ TMP ENDS

# now, convert DMS and control timeseries into pandas timeseries, so we can analyze it
IT_dms_ts = pd.Series(it_DMS_trials_ts)
V1_dms_ts = pd.Series(v1_DMS_trials_ts)
V4_dms_ts = pd.Series(v4_DMS_trials_ts)
D1_dms_ts = pd.Series(d1_DMS_trials_ts)
D2_dms_ts = pd.Series(d2_DMS_trials_ts)
FS_dms_ts = pd.Series(fs_DMS_trials_ts)
FR_dms_ts = pd.Series(fr_DMS_trials_ts)
LIT_dms_ts= pd.Series(lit_DMS_trials_ts)

IT_ctl_ts = pd.Series(it_control_trials_ts)
V1_ctl_ts = pd.Series(v1_control_trials_ts)
V4_ctl_ts = pd.Series(v4_control_trials_ts)
D1_ctl_ts = pd.Series(d1_control_trials_ts)
D2_ctl_ts = pd.Series(d2_control_trials_ts)
FS_ctl_ts = pd.Series(fs_control_trials_ts)
FR_ctl_ts = pd.Series(fr_control_trials_ts)
LIT_ctl_ts= pd.Series(lit_control_trials_ts)

# ... and calculate the functional connectivity of IT with the other modules,
# using the Pearson correlation coefficient
funct_conn_it_v1_dms = IT_dms_ts.corr(V1_dms_ts, method='pearson')
funct_conn_it_v4_dms = IT_dms_ts.corr(V4_dms_ts, method='pearson')
funct_conn_it_d1_dms = IT_dms_ts.corr(D1_dms_ts, method='pearson')
funct_conn_it_d2_dms = IT_dms_ts.corr(D2_dms_ts, method='pearson')
funct_conn_it_fs_dms = IT_dms_ts.corr(FS_dms_ts, method='pearson')
funct_conn_it_fr_dms = IT_dms_ts.corr(FR_dms_ts, method='pearson')
funct_conn_it_lit_dms= IT_dms_ts.corr(LIT_dms_ts,method='pearson')

funct_conn_it_v1_ctl = IT_ctl_ts.corr(V1_ctl_ts, method='pearson')
funct_conn_it_v4_ctl = IT_ctl_ts.corr(V4_ctl_ts, method='pearson')
funct_conn_it_d1_ctl = IT_ctl_ts.corr(D1_ctl_ts, method='pearson')
funct_conn_it_d2_ctl = IT_ctl_ts.corr(D2_ctl_ts, method='pearson')
funct_conn_it_fs_ctl = IT_ctl_ts.corr(FS_ctl_ts, method='pearson')
funct_conn_it_fr_ctl = IT_ctl_ts.corr(FR_ctl_ts, method='pearson')
funct_conn_it_lit_ctl= IT_ctl_ts.corr(LIT_ctl_ts,method='pearson')

# pack correlation coefficients in preparation for saving to a file
func_conn_dms = np.array([funct_conn_it_v1_dms,funct_conn_it_v4_dms,
                          funct_conn_it_fs_dms,funct_conn_it_d1_dms,
                          funct_conn_it_d2_dms,funct_conn_it_fr_dms,
                          funct_conn_it_lit_dms])

func_conn_ctl = np.array([funct_conn_it_v1_ctl,funct_conn_it_v4_ctl,
                          funct_conn_it_fs_ctl,funct_conn_it_d1_ctl,
                          funct_conn_it_d2_ctl,funct_conn_it_fr_ctl,
                          funct_conn_it_lit_ctl])

# now, save all correlation coefficients to a output files 
np.save(func_conn_dms_file, func_conn_dms)
np.save(func_conn_ctl_file, func_conn_ctl)

# define number of groups to plot
N = 2

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([-0.2,1])

# now, group the values to be plotted by brain module
it_v1_corr = (funct_conn_it_v1_dms, funct_conn_it_v1_ctl)
it_v4_corr = (funct_conn_it_v4_dms, funct_conn_it_v4_ctl)
it_fs_corr = (funct_conn_it_fs_dms, funct_conn_it_fs_ctl)
it_d1_corr = (funct_conn_it_d1_dms, funct_conn_it_d1_ctl)
it_d2_corr = (funct_conn_it_d2_dms, funct_conn_it_d2_ctl)
it_fr_corr = (funct_conn_it_fr_dms, funct_conn_it_fr_ctl)
it_lit_corr= (funct_conn_it_lit_dms,funct_conn_it_lit_ctl)

rects_v1 = ax.bar(index, it_v1_corr, width, color='yellow', label='V1')

rects_v4 = ax.bar(index + width, it_v4_corr, width, color='green', label='V4')

rects_fs = ax.bar(index + width*2, it_fs_corr, width, color='orange', label='FS')

rects_d1 = ax.bar(index + width*3, it_d1_corr, width, color='red', label='D1')

rects_d2 = ax.bar(index + width*4, it_d2_corr, width, color='pink', label='D2')

rects_fr = ax.bar(index + width*5, it_fr_corr, width, color='purple', label='FR')

rects_lit= ax.bar(index + width*6, it_lit_corr, width, color='lightblue', label='cIT')

#ax.set_title('FUNCTIONAL CONNECTIVITY OF IT WITH ALL OTHER BRAIN REGIONS')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.set_xlabel('DMS TASK                                        CONTROL TASK')
ax.xaxis.set_label_coords(0.5, -0.025)

ax.set_ylabel('r-value')

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

# Show the plots on the screen
plt.show()
