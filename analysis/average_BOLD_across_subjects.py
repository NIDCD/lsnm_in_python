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
#   This file (average_BOLD_across_subjects.py) was created on April 17, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 11 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# average_BOLD_across_subjects.py
#
# Reads the BOLD timeseries and the synaptic activities from several python (*.npy)
# data files, each corresponding
# to a single subject, and calculates the average timeseries across subjects as well as
# the standard deviation for each data point

import numpy as np
import matplotlib.pyplot as plt

experiment_length = 3960

# scans that were removed from BOLD computation
scans_removed = 0

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# Time for one complete trial in milliseconds
Ttrial = 5.5

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

num_of_scans = T / Tr - scans_removed

num_of_subjects = 10

num_of_modules = 8

# construct array of indices of modules contained in an LSNM model
modules = np.arange(num_of_modules)

# construct array of subjects to be considered
subjects = np.arange(10)

# declare a file name for storing the average BOLD time-series across subjects
avg_syn_file  = 'avg_syn_across_subjs.npy'
avg_BOLD_file = 'avg_BOLD_across_subjs.npy' 

# define the name of the input files where the BOLD and synaptic timeseries are
# stored:
syn_subj = ['subject_11/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_12/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_13/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_14/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_15/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_16/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_17/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_18/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_19/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy',
            'subject_20/output.36trials.with_feedback/synaptic_in_TVB_ROI.npy']
BOLD_subj = ['subject_11/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_12/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_13/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_14/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_15/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_16/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_17/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_18/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_19/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy',
             'subject_20/output.36trials.with_feedback/bold_balloon_TVB_ROI.npy']

# open files that contain synaptic and fMRI BOLD timeseries
lsnm_syn = np.zeros((num_of_subjects, num_of_modules, experiment_length))
lsnm_BOLD = np.zeros((num_of_subjects, num_of_modules, num_of_scans))
for idx in range(len(BOLD_subj)):
    lsnm_syn[idx]  = np.load(syn_subj[idx])
    lsnm_BOLD[idx] = np.load(BOLD_subj[idx])

# construct array indexing scan number of the BOLD timeseries
# (take into account scans that were removed)
total_scans = lsnm_BOLD[0].shape[-1] + scans_removed
total_time = total_scans * Tr
time_removed = scans_removed * Tr
BOLD_timescale = np.arange(time_removed, total_time, Tr)
print BOLD_timescale

# calculate the mean of synaptic and BOLD timeseries across all given subjects
syn_mean = np.mean(lsnm_syn, axis=0)
BOLD_mean = np.mean(lsnm_BOLD, axis=0)

# save the arrays of means to a file for later use
np.save(avg_syn_file,  syn_mean)
np.save(avg_BOLD_file, BOLD_mean)

# calculate the standard deviation of the mean of synaptic and BOLD timeseries across
# subjects
syn_std = np.std(lsnm_syn, axis=0)
BOLD_std = np.std(lsnm_BOLD, axis=0)

# Set up figure to plot fMRI BOLD signal
plt.figure(1)

#plt.suptitle('MEAN SYNAPTIC ACTIVITIES BOLD ACROSS SUBJECTS')

plt.plot(syn_mean[0], linewidth=3.0, color='yellow')
#plt.fill_between(BOLD_timescale, BOLD_mean[0]+BOLD_std[0], BOLD_mean[0]-BOLD_std[0],
#                 facecolor='yellow', alpha=0.1)

plt.gca().set_axis_bgcolor('black')

plt.plot(syn_mean[1], linewidth=3.0, color='lime')

plt.plot(syn_mean[2], linewidth=3.0, color='blue')

plt.plot(syn_mean[3], linewidth=3.0, color='red')

plt.plot(syn_mean[4], linewidth=3.0, color='magenta')

plt.plot(syn_mean[5], linewidth=3.0, color='orange')

plt.plot(syn_mean[6], linewidth=3.0, color='darkorchid')

plt.figure(2)

# increase font size
plt.rcParams.update({'font.size': 30})

#plt.suptitle('MEAN fMRI BOLD SIGNAL ACROSS SUBJECTS')

# plot V1 BOLD time-series in yellow
#plt.plot(BOLD_timescale, 1000 + BOLD_mean[0], linewidth=3.0, color='yellow')
ax = plt.subplot(7,1,1)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
ax.plot(BOLD_timescale, BOLD_mean[0], linewidth=3.0, color='yellow')
#plt.fill_between(BOLD_timescale, BOLD_mean[0]+BOLD_std[0], BOLD_mean[0]-BOLD_std[0],
#                 facecolor='yellow', alpha=0.5)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('V1/V2', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot V4 BOLD time-series in green
ax = plt.subplot(7,1,2)
ax.plot(BOLD_timescale, BOLD_mean[1], linewidth=3.0, color='lime')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
#plt.fill_between(BOLD_timescale, BOLD_mean[1]+BOLD_std[1], BOLD_mean[1]-BOLD_std[1],
#                 facecolor='lime', alpha=0.5)
# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('V4', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot IT BOLD time-series in blue
#plt.plot(BOLD_timescale, 7500 + BOLD_mean[2], linewidth=3.0, color='blue')
ax = plt.subplot(7,1,3)
ax.plot(BOLD_timescale, BOLD_mean[2], linewidth=3.0, color='blue')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
#plt.fill_between(BOLD_timescale, BOLD_mean[2]+BOLD_std[2], BOLD_mean[2]-BOLD_std[2],
#                 facecolor='blue', alpha=0.5)
# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot FS BOLD time-series in orange
#plt.plot(BOLD_timescale, 2500 + BOLD_mean[3], linewidth=3.0, color='orange')
ax = plt.subplot(7,1,4)
ax.plot(BOLD_timescale, BOLD_mean[3], linewidth=3.0, color='orange')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
#plt.fill_between(BOLD_timescale, BOLD_mean[3]+BOLD_std[3], BOLD_mean[3]-BOLD_std[3],
#                 facecolor='orange', alpha=0.5)
# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot D1 BOLD time-series in red
#plt.plot(BOLD_timescale, 2000 + BOLD_mean[4], linewidth=3.0, color='red')
ax = plt.subplot(7,1,5)
ax.plot(BOLD_timescale, BOLD_mean[4], linewidth=3.0, color='red')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
#plt.fill_between(BOLD_timescale, BOLD_mean[4]+BOLD_std[4], BOLD_mean[4]-BOLD_std[4],
#                 facecolor='red', alpha=0.5)
# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot D2 BOLD time-series in pink
#plt.plot(BOLD_timescale, -1500 + BOLD_mean[5], linewidth=3.0, color='pink')
ax = plt.subplot(7,1,6)
ax.plot(BOLD_timescale, BOLD_mean[5], linewidth=3.0, color='pink')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
#plt.fill_between(BOLD_timescale, BOLD_mean[5]+BOLD_std[5], BOLD_mean[5]-BOLD_std[5],
#                 facecolor='pink', alpha=0.5)
# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot FR BOLD time-series in purple
#plt.plot(BOLD_timescale, -500 + BOLD_mean[6], linewidth=3.0, color='darkorchid')
ax = plt.subplot(7,1,7)
ax.plot(BOLD_timescale, BOLD_mean[6], linewidth=3.0, color='darkorchid')
ax.set_yticks([])
ax.set_xlim(0,200)
#plt.fill_between(BOLD_timescale, BOLD_mean[6]+BOLD_std[6], BOLD_mean[6]-BOLD_std[6],
#                 facecolor='darkorchid', alpha=0.5)
# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# Show the plots on the screen
plt.show()
