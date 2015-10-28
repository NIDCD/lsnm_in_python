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
#   This file (compute_rCBF.py) was created on September 29, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 29 2015
#
# **************************************************************************/

# compute_rCBF.py
#
# Calculate rCBF, separately, for hi attention and low attention trials 
# ... using data from visual delay-match-to-sample simulation.
# It also print the rCBF values for each and all modules, and for high
# and low attention levels 
#
# The input data (synaptic activities) are numpy arrays
# with columns in the following order:
#
# V1 ROI (right hemisphere, includes LSNM units and TVB nodes) 
# V4 ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (right hemisphere, includes LSNM units and TVB nodes)
# FS ROI (right hemisphere, includes LSNM units and TVB nodes)
# D1 ROI (right hemisphere, includes LSNM units and TVB nodes)
# D2 ROI (right hemisphere, includes LSNM units and TVB nodes)
# FR ROI (right hemisphere, includes LSNM units and TVB nodes)

import numpy as np

import matplotlib.pyplot as plt

# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_ROI.npy'

# define an array with location of low-attention blocks, and another array
# with location of hi-attention blocks, relative to
# an array that contains all blocks
control_trials = [3,4,5,9,10,11,15,16,17,21,22,23,27,28,29,33,34,35]
dms_trials =     [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32]

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110
number_of_trials = 36

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 5 ms.
Ti = 0.005 * 10

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# Time for one complete trial in milliseconds
Ttrial = 5.5

# read the input file that contains the synaptic activities of all ROIs
syn = np.load(SYN_file)

# extract the synaptic activities corresponding to each ROI:
v1_syn = syn[0]
v4_syn = syn[1]
it_syn = syn[2]
fs_syn = syn[3]
d1_syn = syn[4]
d2_syn = syn[5]
fr_syn = syn[6]

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = v1_syn.size

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

# but also, get rid of the DMS task trials, to create arrays that contain only control trials
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

# ... consolidate all prefrontal areas into a single one:
pf_DMS_trials_ts = fs_DMS_trials_ts + d1_DMS_trials_ts + \
                   d2_DMS_trials_ts + fr_DMS_trials_ts

pf_control_trials_ts = fs_control_trials_ts + d1_control_trials_ts + \
                       d2_control_trials_ts + fr_control_trials_ts

# ... and now, let's integrate synaptic activities across time, and normalize to the value of
# the high attention in V1:
v1_rCBF_hiatt = np.sum(v1_DMS_trials_ts)
v4_rCBF_hiatt = np.sum(v4_DMS_trials_ts) / v1_rCBF_hiatt
it_rCBF_hiatt = np.sum(it_DMS_trials_ts) / v1_rCBF_hiatt
pf_rCBF_hiatt = np.sum(pf_DMS_trials_ts) / v1_rCBF_hiatt

v1_rCBF_loatt = np.sum(v1_control_trials_ts) / v1_rCBF_hiatt
v4_rCBF_loatt = np.sum(v4_control_trials_ts) / v1_rCBF_hiatt
it_rCBF_loatt = np.sum(it_control_trials_ts) / v1_rCBF_hiatt
pf_rCBF_loatt = np.sum(pf_control_trials_ts) / v1_rCBF_hiatt

# ...and, finally, caculate percentage change from high att to low att:
v1_rCBF_pc_change = (1.0 - v1_rCBF_loatt) * 100. / 1.0
v4_rCBF_pc_change = (v4_rCBF_hiatt - v4_rCBF_loatt) * 100. / v4_rCBF_hiatt
it_rCBF_pc_change = (it_rCBF_hiatt - it_rCBF_loatt) * 100. / it_rCBF_hiatt
pf_rCBF_pc_change = (pf_rCBF_hiatt - pf_rCBF_loatt) * 100. / pf_rCBF_hiatt

# print out the results. No need to save to a file as it is a rather small table
print 'V1 rCBF (hi, lo, percent change): ', 1.0, v1_rCBF_loatt, v1_rCBF_pc_change
print 'V4 rCBF (hi, lo, percent change): ', v4_rCBF_hiatt, v4_rCBF_loatt, v4_rCBF_pc_change
print 'IT rCBF (hi, lo, percent change): ', it_rCBF_hiatt, it_rCBF_loatt, it_rCBF_pc_change
print 'PF rCBF (hi, lo, percent change): ', pf_rCBF_hiatt, pf_rCBF_loatt, pf_rCBF_pc_change
