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
#   This file (compute_avg_MIs.py) was created on April 26, 2016.
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on April 27, 2016  
# **************************************************************************/

# compute_avg_MIs.py
#
# Calculate and plot mean and individual modulation effects for the auditory DMS
# simulations using previously computed Modulation Indexes (MI) for each
# simulated subject and for each simulated condition and trial. The MIs were previously
# computed and are contained within subject/condition directories.
#

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from scipy import stats

# set matplot lib parameters to produce visually appealing plots
#mpl.style.use('ggplot')

# increase font size prior to plotting
plt.rcParams.update({'font.size': 15})

# define number of groups to plot
N = 10

num_of_trials = 12

num_of_subjects = 10

# define the names of the input files where the MIs are contained, for four conditions:
# TC-PSL, Tones-PSL, TC-DMS and Tones-DMS
MI_TC_PSL_subj = np.array(['subject_original_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_2_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_3_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_4_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_5_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_6_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_7_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_8_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_9_with_feedback/output.TC_PSL/modulation_index.npy',
                            'subject_10_with_feedback/output.TC_PSL/modulation_index.npy'])

MI_Tones_PSL_subj = np.array(['subject_original_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_2_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_3_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_4_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_5_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_6_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_7_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_8_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_9_with_feedback/output.Tones_PSL/modulation_index.npy',
                            'subject_10_with_feedback/output.Tones_PSL/modulation_index.npy'])

MI_TC_DMS_subj = np.array(['subject_original_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_2_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_3_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_4_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_5_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_6_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_7_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_8_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_9_with_feedback/output.TC_DMS/modulation_index.npy',
                            'subject_10_with_feedback/output.TC_DMS/modulation_index.npy'])

MI_Tones_DMS_subj = np.array(['subject_original_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_2_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_3_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_4_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_5_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_6_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_7_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_8_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_9_with_feedback/output.Tones_DMS/modulation_index.npy',
                            'subject_10_with_feedback/output.Tones_DMS/modulation_index.npy'])
                           
# open files containing MIs for four conditions: TC-PSL, Tones-PSL, TC-DMS and Tones-DMS
MI_TC_PSL = np.zeros((num_of_subjects, num_of_trials))
for idx in range(0, num_of_subjects):
    MI_TC_PSL[idx]  =  np.load(MI_TC_PSL_subj[idx])
                              
MI_Tones_PSL = np.zeros((num_of_subjects, num_of_trials))
for idx in range(0, num_of_subjects):
    MI_Tones_PSL[idx]= np.load(MI_Tones_PSL_subj[idx])
                              
MI_TC_DMS = np.zeros((num_of_subjects, num_of_trials))
for idx in range(0, num_of_subjects):
    MI_TC_DMS[idx]  =  np.load(MI_TC_DMS_subj[idx])
                              
MI_Tones_DMS = np.zeros((num_of_subjects, num_of_trials))
for idx in range(0, num_of_subjects):
    MI_Tones_DMS[idx]= np.load(MI_Tones_DMS_subj[idx])

# Calculate the mean (and std) of modulation effects across trials for the TC-DMS and the
# TC-PSL conditions for each subject
mean_MI_TC_PSL = np.zeros(num_of_subjects)
std_MI_TC_PSL  = np.zeros(num_of_subjects)
mean_MI_TC_DMS = np.zeros(num_of_subjects)
std_MI_TC_DMS  = np.zeros(num_of_subjects)
for idx in range(0, num_of_subjects):
    mean_MI_TC_PSL[idx] = np.mean(MI_TC_PSL[idx])
    std_MI_TC_PSL[idx]  = np.std(MI_TC_PSL[idx])
    mean_MI_TC_DMS[idx] = np.mean(MI_TC_DMS[idx])
    std_MI_TC_DMS[idx]  = np.std(MI_TC_DMS[idx])
    
# Calculate the mean modulation effects for each task condition, keeing match and nonmatch separate,
# across all subjects
mean_MI_TC_PSL_match       = np.mean(MI_TC_PSL[:,0::2])
sem_MI_TC_PSL_match        = np.std(MI_TC_PSL[:,0::2])

mean_MI_TC_PSL_nonmatch    = np.mean(MI_TC_PSL[:,1::2])
sem_MI_TC_PSL_nonmatch     = np.std(MI_TC_PSL[:,1::2])

mean_MI_Tones_PSL_match    = np.mean(MI_Tones_PSL[:,0::2])
sem_MI_Tones_PSL_match     = np.std(MI_Tones_PSL[:,0::2])

mean_MI_Tones_PSL_nonmatch = np.mean(MI_Tones_PSL[:,1::2])
sem_MI_Tones_PSL_nonmatch  = np.std(MI_Tones_PSL[:,1::2])

mean_MI_TC_DMS_match       = np.mean(MI_TC_DMS[:,0::2])
sem_MI_TC_DMS_match        = np.std(MI_TC_DMS[:,0::2])

mean_MI_TC_DMS_nonmatch    = np.mean(MI_TC_DMS[:,1::2])
sem_MI_TC_DMS_nonmatch     = np.std(MI_TC_DMS[:,1::2])

mean_MI_Tones_DMS_match    = np.mean(MI_Tones_DMS[:,0::2])
sem_MI_Tones_DMS_match     = np.std(MI_Tones_DMS[:,0::2])

mean_MI_Tones_DMS_nonmatch = np.mean(MI_Tones_DMS[:,1::2])
sem_MI_Tones_DMS_nonmatch  = np.std(MI_Tones_DMS[:,1::2])

    
# First figure to contain modulations indexes for TC across trials for each simulated
# subject
# create a list of x locations for each group 
index = np.arange(N)            
width = 0.4                     # width of the bars

fig, ax = plt.subplots()

#ax.set_ylim([0,50])

# now, group the values to be plotted by brain module and by task condition

rects_tc_psl = ax.bar(index, mean_MI_TC_PSL, width, color='steelblue', yerr=std_MI_TC_PSL)
rects_tc_dms = ax.bar(index + width, mean_MI_TC_DMS, width, color='darkred', yerr=std_MI_TC_DMS)

ax.set_title('Individual Modulation Effects')

ax.set_xticks(index + width)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))

ax.set_ylabel('Modulation Index (%)')

ax.set_xlabel('Simulated Subjects')

ax.legend((rects_tc_psl[0], rects_tc_dms[0]), ('TC_PSL', 'TC_DMS'))


# Second figure to contain modulation indexes for Tones and TC, match and nonmatch
# separately, across all simulated subjects
# we will have two groups here, PSL and DMS
N=2

index = np.arange(N)            
width = 0.1                     # width of the bars

fig, ax = plt.subplots()

ax.set_ylim([0,40])

# now, group the values to be plotted by task condition (PSL and DMS)

rects_tones_match    = ax.bar(index, (mean_MI_Tones_PSL_match, mean_MI_Tones_DMS_match),
                              width, color='steelblue', align='center',
                              yerr=(sem_MI_Tones_PSL_match, sem_MI_Tones_DMS_match),
                              ecolor= 'k')

rects_tones_nonmatch = ax.bar(index + width, (mean_MI_Tones_PSL_nonmatch, mean_MI_Tones_DMS_nonmatch),
                              width, color='darkred', align='center',
                              yerr=(sem_MI_Tones_PSL_nonmatch, sem_MI_Tones_DMS_nonmatch),
                              ecolor= 'k')

rects_tc_match       = ax.bar(index + width*2, (mean_MI_TC_PSL_match, mean_MI_TC_DMS_match),
                              width, color='lightyellow', align='center',
                              yerr=(sem_MI_TC_PSL_match, sem_MI_TC_DMS_match),
                              ecolor= 'k')

rects_tc_nonmatch    = ax.bar(index + width*3, (mean_MI_TC_PSL_nonmatch, mean_MI_TC_DMS_nonmatch),
                              width, color='lightblue', align='center',
                              yerr=(sem_MI_TC_PSL_nonmatch, sem_MI_TC_DMS_nonmatch),
                              ecolor= 'k')


ax.set_title('Mean Modulation Effects')

ax.set_xticks(index + width)
ax.set_xticklabels(('PSL', 'DMS'))

ax.set_ylabel('Modulation Index (%)')

ax.set_xlabel('Task')

ax.legend((rects_tones_match[0], rects_tones_nonmatch[0], rects_tc_match[0], rects_tc_nonmatch[0]),
          ('Tone_Match', 'Tone_nonMatch', 'TC_Match', 'TC_nonMatch'))



# Show the plots on the screen
plt.show()

