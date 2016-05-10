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
#   This file (func_conn_syn_tvb.py) was created on May 4, 2015.
#
#   Based in part by Matlab scripts by Horwitz et al.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 7 2015  
# **************************************************************************/

# func_conn_syn_tvb.py
#
# Calculate and plot functional connectivity (within-task time series correlation)
# of IT with all other simulated brain areas, using the output 
# from Hagmann's brain (integrated synaptic activity) WITHOUT LSNM modules.

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_ROI_tvb.npy'

# define the name of the output file where the functional connectivity timeseries will be stored
func_conn_dms_file = 'corr_syn_IT_vs_all_tvb.npy'

# define the length of both each trial and the whole experiment
# in synaptic timesteps, as well as total number of trials
experiment_length = 3960
trial_length = 110
number_of_trials = 36

# define intertrial interval duration in number of synaptic timesteps
#ITI_length = 20

# open file that contains the synaptic activities
syn = np.load(SYN_file)

# extract synaptic activities for each ROI
v1_syn = syn[0, 1:]
v4_syn = syn[1, 1:]
it_syn = syn[2, 1:]
fs_syn = syn[3, 1:]
d1_syn = syn[4, 1:]
d2_syn = syn[5, 1:]
fr_syn = syn[6, 1:]
lit_syn= syn[7, 1:]

# now, convert DMS and control timeseries into pandas timeseries, so we can analyze it
IT_dms_ts = pd.Series(it_syn)
V1_dms_ts = pd.Series(v1_syn)
V4_dms_ts = pd.Series(v4_syn)
D1_dms_ts = pd.Series(d1_syn)
D2_dms_ts = pd.Series(d2_syn)
FS_dms_ts = pd.Series(fs_syn)
FR_dms_ts = pd.Series(fr_syn)
LIT_dms_ts= pd.Series(lit_syn)

# ... and calculate the functional connectivity of IT with the other modules,
# using the Pearson correlation coefficient
funct_conn_it_v1_dms = IT_dms_ts.corr(V1_dms_ts, method='pearson')
funct_conn_it_v4_dms = IT_dms_ts.corr(V4_dms_ts, method='pearson')
funct_conn_it_d1_dms = IT_dms_ts.corr(D1_dms_ts, method='pearson')
funct_conn_it_d2_dms = IT_dms_ts.corr(D2_dms_ts, method='pearson')
funct_conn_it_fs_dms = IT_dms_ts.corr(FS_dms_ts, method='pearson')
funct_conn_it_fr_dms = IT_dms_ts.corr(FR_dms_ts, method='pearson')
funct_conn_it_lit_dms= IT_dms_ts.corr(LIT_dms_ts,method='pearson')

# pack correlation coefficients in preparation for saving to a file
func_conn_dms = np.array([funct_conn_it_v1_dms,funct_conn_it_v4_dms,
                          funct_conn_it_fs_dms,funct_conn_it_d1_dms,
                          funct_conn_it_d2_dms,funct_conn_it_fr_dms,
                          funct_conn_it_lit_dms])

# now, save all correlation coefficients to a output files 
np.save(func_conn_dms_file, func_conn_dms)

# define number of groups to plot
N = 1

# create a list of x locations for each group 
index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([-0.4,1])

rects_v1 = ax.bar(index, funct_conn_it_v1_dms, width, color='purple', label='V1')

rects_v4 = ax.bar(index + width, funct_conn_it_v4_dms, width, color='darkred', label='V4')

rects_fs = ax.bar(index + width*2, funct_conn_it_fs_dms, width, color='lightyellow', label='FS')

rects_d1 = ax.bar(index + width*3, funct_conn_it_d1_dms, width, color='lightblue', label='D1')

rects_d2 = ax.bar(index + width*4, funct_conn_it_d2_dms, width, color='yellow', label='D2')

rects_fr = ax.bar(index + width*5, funct_conn_it_fr_dms, width, color='red', label='FR')

rects_lit= ax.bar(index + width*6, funct_conn_it_lit_dms, width, color='green', label='left IT')

ax.set_title('FUNCTIONAL CONNECTIVITY OF IT WITH ALL OTHER BRAIN REGIONS')

# get rid of x axis ticks and labels
ax.set_xticks([])

ax.xaxis.set_label_coords(0.5, -0.025)

# Shrink current axis by 10% to make space for legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# place a legend to the right of the figure
plt.legend(loc='center left', bbox_to_anchor=(1.02, .5))

# Show the plots on the screen
plt.show()
