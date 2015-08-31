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
# Reads the BOLD timeseries from several python (*.npy) data files, each corresponding
# to a single subject, and calculates the average timeseries across subjects as well as
# the standard deviation for each data point

import numpy as np
import matplotlib.pyplot as plt

# scans that were removed from BOLD computation
#scans_removed = 4
scans_removed = 7

# what is the TR used in simulated BOLD
TR = 2.0

# construct array of indices of modules contained in an LSNM model
modules = np.arange(7)

# construct array of subjects to be considered
subjects = np.arange(10)

# define the name of the output file where the BOLD timeseries will be stored
BOLD_subj1 = '../visual_model/subject_1/output.36trials/lsnm_bold.npy'
BOLD_subj2 = '../visual_model/subject_2/output.36trials/lsnm_bold.npy'
BOLD_subj3 = '../visual_model/subject_3/output.36trials/lsnm_bold.npy'
BOLD_subj4 = '../visual_model/subject_4/output.36trials/lsnm_bold.npy'
BOLD_subj5 = '../visual_model/subject_5/output.36trials/lsnm_bold.npy'
BOLD_subj6 = '../visual_model/subject_6/output.36trials/lsnm_bold.npy'
BOLD_subj7 = '../visual_model/subject_7/output.36trials/lsnm_bold.npy'
BOLD_subj8 = '../visual_model/subject_8/output.36trials/lsnm_bold.npy'
BOLD_subj9 = '../visual_model/subject_9/output.36trials/lsnm_bold.npy'
BOLD_subj10 = '../visual_model/subject_10/output.36trials/lsnm_bold.npy'


# open files that contain fMRI BOLD timeseries
lsnm_BOLD_subj1 = np.load(BOLD_subj1)
lsnm_BOLD_subj2 = np.load(BOLD_subj2)
lsnm_BOLD_subj3 = np.load(BOLD_subj3)
lsnm_BOLD_subj4 = np.load(BOLD_subj4)
lsnm_BOLD_subj5 = np.load(BOLD_subj5)
lsnm_BOLD_subj6 = np.load(BOLD_subj6)
lsnm_BOLD_subj7 = np.load(BOLD_subj7)
lsnm_BOLD_subj8 = np.load(BOLD_subj8)
lsnm_BOLD_subj9 = np.load(BOLD_subj9)
lsnm_BOLD_subj10 = np.load(BOLD_subj10)

# construct a numpy array that contains the BOLD timeseries for all subjects
lsnm_BOLD = np.array([lsnm_BOLD_subj1, lsnm_BOLD_subj2, lsnm_BOLD_subj3,
                      lsnm_BOLD_subj4, lsnm_BOLD_subj5, lsnm_BOLD_subj6,
                      lsnm_BOLD_subj7, lsnm_BOLD_subj8, lsnm_BOLD_subj9, lsnm_BOLD_subj10 ]) 

# construct array indexing scan number of the BOLD timeseries
# (take into account scans that were removed)
total_scans = lsnm_BOLD_subj1.shape[-1] + scans_removed
total_time = total_scans * TR
time_removed = scans_removed * TR
BOLD_timescale = np.arange(time_removed, total_time, TR)
print BOLD_timescale

# calculate the mean of BOLD timeseries across all given subjects
BOLD_mean = np.mean(lsnm_BOLD, axis=0)

# calculate the standard deviation of the mean of BOLD timeseries across subjects
BOLD_std = np.std(lsnm_BOLD, axis=0)

# Set up separate figures to plot fMRI BOLD signal
plt.figure(1)

plt.suptitle('MEAN fMRI BOLD SIGNAL ACROSS SUBJECTS')

plt.plot(BOLD_timescale, 13100 + BOLD_mean[0], linewidth=3.0, color='yellow')
#plt.fill_between(BOLD_ts_length, BOLD_mean[0]+BOLD_std[0], BOLD_mean[0]-BOLD_std[0],
#                 facecolor='yellow', alpha=0.1)
plt.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
plt.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
plt.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
plt.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
plt.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
plt.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.gca().set_axis_bgcolor('black')

#plt.figure(2)

#plt.suptitle('MEAN fMRI BOLD SIGNAL IN V4 ACROSS SUBJECTS')

plt.plot(BOLD_timescale, -12500 + BOLD_mean[1], linewidth=3.0, color='lime')
#plt.fill_between(BOLD_ts_length, BOLD_mean[1]+BOLD_std[1], BOLD_mean[1]-BOLD_std[1],
#                 facecolor='green', alpha=0.1)
#plt.gca().set_axis_bgcolor('black')

#plt.figure(3)
#plt.suptitle('MEAN fMRI BOLD SIGNAL IN IT ACROSS SUBJECTS')

plt.plot(BOLD_timescale, 32000 + BOLD_mean[2], linewidth=3.0, color='blue')
#plt.fill_between(BOLD_ts_length, BOLD_mean[2]+BOLD_std[2], BOLD_mean[2]-BOLD_std[2],
#                 facecolor='blue', alpha=0.1)
#plt.gca().set_axis_bgcolor('black')

#plt.figure(4)
#plt.suptitle('MEAN fMRI BOLD SIGNAL IN D1 ACROSS SUBJECTS')

plt.plot(BOLD_timescale,  -14600 + BOLD_mean[3], linewidth=3.0, color='red')
#plt.fill_between(BOLD_ts_length, BOLD_mean[3]+BOLD_std[3], BOLD_mean[3]-BOLD_std[3],
#                 facecolor='red', alpha=0.1)
#plt.gca().set_axis_bgcolor('black')

#plt.figure(5)
#plt.suptitle('MEAN fMRI BOLD SIGNAL IN D2 ACROSS SUBJECTS')

plt.plot(BOLD_timescale, 17400 + BOLD_mean[4], linewidth=3.0, color='magenta')
#plt.fill_between(BOLD_ts_length, BOLD_mean[4]+BOLD_std[4], BOLD_mean[4]-BOLD_std[4],
#                 facecolor='magenta', alpha=0.1)
#plt.gca().set_axis_bgcolor('black')

#plt.figure(6)
#plt.suptitle('MEAN fMRI BOLD SIGNAL IN FS ACROSS SUBJECTS')

plt.plot(BOLD_timescale, 11000 + BOLD_mean[5], linewidth=3.0, color='orange')
#plt.fill_between(BOLD_ts_length, BOLD_mean[5]+BOLD_std[5], BOLD_mean[5]-BOLD_std[5],
#                 facecolor='orange', alpha=0.1)
#plt.gca().set_axis_bgcolor('black')

#plt.figure(7)
#plt.suptitle('MEAN fMRI BOLD SIGNAL IN FR ACROSS SUBJECTS')

plt.plot(BOLD_timescale, 1000 + BOLD_mean[6], linewidth=3.0, color='darkorchid')
#plt.fill_between(BOLD_ts_length, BOLD_mean[6]+BOLD_std[6], BOLD_mean[6]-BOLD_std[6],
#                 facecolor='purple', alpha=0.1)
#plt.gca().set_axis_bgcolor('black')


# Show the plots on the screen
plt.show()
