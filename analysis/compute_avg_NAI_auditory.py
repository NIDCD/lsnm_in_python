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
#   This file (compute_avg_NAI_auditory.py) was created on April 19, 2016.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on April 19, 2016  
# **************************************************************************/

# compute_avg_NAI_auditory.py
#
# Compute average of Neural Activity Index (NAI) of auditory primary cortex using NAI's for a 
# a number of subjects (previously stored in a python data file)

import numpy as np
import matplotlib.pyplot as plt

num_of_subjects = 10
NAI_ts_length = 30

# define the names of the input files where the NAI timecourses are contained, for four conditions:
# TC-PSL, Tones-PSL, TC-DMS and Tones-DMS
NAI_TC_PSL_subj = np.array(['subject_original_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_2_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_3_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_4_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_5_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_6_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_7_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_8_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_9_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                            'subject_10_with_feedback/output.TC_PSL/NAI_timecourse.npy',
                        ])
NAI_Tones_PSL_subj = np.array(['subject_original_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_2_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_3_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_4_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_5_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_6_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_7_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_8_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_9_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                               'subject_10_with_feedback/output.Tones_PSL/NAI_timecourse.npy',
                           ])
NAI_TC_DMS_subj = np.array(['subject_original_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_2_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_3_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_4_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_5_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_6_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_7_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_8_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_9_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                            'subject_10_with_feedback/output.TC_DMS/NAI_timecourse.npy',
                        ])
NAI_Tones_DMS_subj = np.array(['subject_original_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_2_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_3_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_4_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_5_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_6_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_7_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_8_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_9_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                               'subject_10_with_feedback/output.Tones_DMS/NAI_timecourse.npy',
                           ])



# Load NAI source activity data files
# open files containing NAI time-series
NAI_TC_PSL = np.zeros((num_of_subjects, 2, NAI_ts_length))
NAI_Tones_PSL = np.zeros((num_of_subjects, 2, NAI_ts_length))
NAI_TC_DMS = np.zeros((num_of_subjects, 2, NAI_ts_length))
NAI_Tones_DMS = np.zeros((num_of_subjects, 2, NAI_ts_length))
for idx in range(0, num_of_subjects):
    NAI_TC_PSL[idx] = np.load(NAI_TC_PSL_subj[idx])    
    NAI_Tones_PSL[idx] = np.load(NAI_Tones_PSL_subj[idx])    
    NAI_TC_DMS[idx] = np.load(NAI_TC_DMS_subj[idx])    
    NAI_Tones_DMS[idx] = np.load(NAI_Tones_DMS_subj[idx])    

# average NAI across subjects
for idx in range(0, num_of_subjects):
    NAI_TC_PSL_avg = np.mean(NAI_TC_PSL, axis=0)
    NAI_Tones_PSL_avg = np.mean(NAI_Tones_PSL, axis=0)
    NAI_TC_DMS_avg = np.mean(NAI_TC_DMS, axis=0)
    NAI_Tones_DMS_avg = np.mean(NAI_Tones_DMS, axis=0)

# plot the average NAI's

# Set up figure to plot MEG source dynamics averaged across trials
fig = plt.figure(1)

plt.suptitle('NAI average timecourse during TC DMS')

# Plot NAI
aud_plot_S1 = plt.plot(NAI_TC_DMS_avg[0], label='S1')
aud_plot_S2 = plt.plot(NAI_TC_DMS_avg[1], label='S2')

plt.legend()

# Set up figure to plot MEG source dynamics averaged across trials
fig = plt.figure(2)

plt.suptitle('NAI average timecourse during Tones DMS')

# Plot NAI
aud_plot_S1 = plt.plot(NAI_Tones_DMS_avg[0], label='S1')
aud_plot_S2 = plt.plot(NAI_Tones_DMS_avg[1], label='S2')

plt.legend()
# Set up figure to plot MEG source dynamics averaged across trials
fig = plt.figure(3)

plt.suptitle('NAI average timecourse during TC PSL')

# Plot NAI
aud_plot_S1 = plt.plot(NAI_TC_PSL_avg[0], label='S1')
aud_plot_S2 = plt.plot(NAI_TC_PSL_avg[1], label='S2')

plt.legend()
# Set up figure to plot MEG source dynamics averaged across trials
fig = plt.figure(4)

plt.suptitle('NAI average timecourse during Tones PSL')

# Plot NAI
aud_plot_S1 = plt.plot(NAI_Tones_PSL_avg[0], label='S1')
aud_plot_S2 = plt.plot(NAI_Tones_PSL_avg[1], label='S2')

plt.legend()

# Show the plot on the screen
plt.show()
