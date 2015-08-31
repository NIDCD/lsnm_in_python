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
#   This file (avg_func_conn_across_subjects.py) was created on August 23, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 11 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# avg_func_conn_across_subjects.py
#
# Reads the correlation coefficients from several python (*.npy) data files, each
# corresponding to a single subject, and calculates the average functional connectivity
# across all subjects, as well as the standard deviation for each data point. It uses
# previously calculated correlation coefficients between IT and all other areas in both,
# synapctic activity time-series, and fMRI bold time-series.

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# construct array of indices of modules contained in an LSNM model, minus 1
modules = np.arange(6)

# construct array of subjects to be considered
subjects = np.arange(10)

# define the names of the output files where the correlation coefficients were stored
func_conn_syn_dms_subj1 = '../visual_model/subject_1/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj2 = '../visual_model/subject_2/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj3 = '../visual_model/subject_3/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj4 = '../visual_model/subject_4/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj5 = '../visual_model/subject_5/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj6 = '../visual_model/subject_6/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj7 = '../visual_model/subject_7/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj8 = '../visual_model/subject_8/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj9 = '../visual_model/subject_9/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj10 = '../visual_model/subject_10/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_ctl_subj1 = '../visual_model/subject_1/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj2 = '../visual_model/subject_2/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj3 = '../visual_model/subject_3/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj4 = '../visual_model/subject_4/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj5 = '../visual_model/subject_5/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj6 = '../visual_model/subject_6/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj7 = '../visual_model/subject_7/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj8 = '../visual_model/subject_8/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj9 = '../visual_model/subject_9/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj10 = '../visual_model/subject_10/output.36trials/corr_syn_IT_vs_all_ctl.npy'
func_conn_fmri_dms_subj1 = '../visual_model/subject_1/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj2 = '../visual_model/subject_2/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj3 = '../visual_model/subject_3/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj4 = '../visual_model/subject_4/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj5 = '../visual_model/subject_5/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj6 = '../visual_model/subject_6/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj7 = '../visual_model/subject_7/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj8 = '../visual_model/subject_8/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj9 = '../visual_model/subject_9/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_dms_subj10 = '../visual_model/subject_10/output.36trials/corr_fmri_IT_vs_all_dms.npy'
func_conn_fmri_ctl_subj1 = '../visual_model/subject_1/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj2 = '../visual_model/subject_2/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj3 = '../visual_model/subject_3/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj4 = '../visual_model/subject_4/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj5 = '../visual_model/subject_5/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj6 = '../visual_model/subject_6/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj7 = '../visual_model/subject_7/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj8 = '../visual_model/subject_8/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj9 = '../visual_model/subject_9/output.36trials/corr_fmri_IT_vs_all_ctl.npy'
func_conn_fmri_ctl_subj10 = '../visual_model/subject_10/output.36trials/corr_fmri_IT_vs_all_ctl.npy'

# open files that contain correlation coefficients
fc_syn_dms_subj1 = np.load(func_conn_syn_dms_subj1)
fc_syn_dms_subj2 = np.load(func_conn_syn_dms_subj2)
fc_syn_dms_subj3 = np.load(func_conn_syn_dms_subj3)
fc_syn_dms_subj4 = np.load(func_conn_syn_dms_subj4)
fc_syn_dms_subj5 = np.load(func_conn_syn_dms_subj5)
fc_syn_dms_subj6 = np.load(func_conn_syn_dms_subj6)
fc_syn_dms_subj7 = np.load(func_conn_syn_dms_subj7)
fc_syn_dms_subj8 = np.load(func_conn_syn_dms_subj8)
fc_syn_dms_subj9 = np.load(func_conn_syn_dms_subj9)
fc_syn_dms_subj10 = np.load(func_conn_syn_dms_subj10)
fc_syn_ctl_subj1 = np.load(func_conn_syn_ctl_subj1)
fc_syn_ctl_subj2 = np.load(func_conn_syn_ctl_subj2)
fc_syn_ctl_subj3 = np.load(func_conn_syn_ctl_subj3)
fc_syn_ctl_subj4 = np.load(func_conn_syn_ctl_subj4)
fc_syn_ctl_subj5 = np.load(func_conn_syn_ctl_subj5)
fc_syn_ctl_subj6 = np.load(func_conn_syn_ctl_subj6)
fc_syn_ctl_subj7 = np.load(func_conn_syn_ctl_subj7)
fc_syn_ctl_subj8 = np.load(func_conn_syn_ctl_subj8)
fc_syn_ctl_subj9 = np.load(func_conn_syn_ctl_subj9)
fc_syn_ctl_subj10 = np.load(func_conn_syn_ctl_subj10)
fc_fmri_dms_subj1 = np.load(func_conn_fmri_dms_subj1)
fc_fmri_dms_subj2 = np.load(func_conn_fmri_dms_subj2)
fc_fmri_dms_subj3 = np.load(func_conn_fmri_dms_subj3)
fc_fmri_dms_subj4 = np.load(func_conn_fmri_dms_subj4)
fc_fmri_dms_subj5 = np.load(func_conn_fmri_dms_subj5)
fc_fmri_dms_subj6 = np.load(func_conn_fmri_dms_subj6)
fc_fmri_dms_subj7 = np.load(func_conn_fmri_dms_subj7)
fc_fmri_dms_subj8 = np.load(func_conn_fmri_dms_subj8)
fc_fmri_dms_subj9 = np.load(func_conn_fmri_dms_subj9)
fc_fmri_dms_subj10 = np.load(func_conn_fmri_dms_subj10)
fc_fmri_ctl_subj1 = np.load(func_conn_fmri_ctl_subj1)
fc_fmri_ctl_subj2 = np.load(func_conn_fmri_ctl_subj2)
fc_fmri_ctl_subj3 = np.load(func_conn_fmri_ctl_subj3)
fc_fmri_ctl_subj4 = np.load(func_conn_fmri_ctl_subj4)
fc_fmri_ctl_subj5 = np.load(func_conn_fmri_ctl_subj5)
fc_fmri_ctl_subj6 = np.load(func_conn_fmri_ctl_subj6)
fc_fmri_ctl_subj7 = np.load(func_conn_fmri_ctl_subj7)
fc_fmri_ctl_subj8 = np.load(func_conn_fmri_ctl_subj8)
fc_fmri_ctl_subj9 = np.load(func_conn_fmri_ctl_subj9)
fc_fmri_ctl_subj10 = np.load(func_conn_fmri_ctl_subj10)

# construct numpy arrays that contain correlation coefficients for all subjects
# (the functional connectivity of IT versus the other 6 modules of LSNM model)
fc_syn_dms = np.array([fc_syn_dms_subj1, fc_syn_dms_subj2, fc_syn_dms_subj3,
                       fc_syn_dms_subj4, fc_syn_dms_subj5, fc_syn_dms_subj6,
                       fc_syn_dms_subj7, fc_syn_dms_subj8, fc_syn_dms_subj9,
                       fc_syn_dms_subj10 ]) 
fc_syn_ctl = np.array([fc_syn_ctl_subj1, fc_syn_ctl_subj2, fc_syn_ctl_subj3,
                       fc_syn_ctl_subj4, fc_syn_ctl_subj5, fc_syn_ctl_subj6,
                       fc_syn_ctl_subj7, fc_syn_ctl_subj8, fc_syn_ctl_subj9,
                       fc_syn_ctl_subj10 ]) 
fc_fmri_dms = np.array([fc_fmri_dms_subj1, fc_fmri_dms_subj2, fc_fmri_dms_subj3,
                        fc_fmri_dms_subj4, fc_fmri_dms_subj5, fc_fmri_dms_subj6,
                        fc_fmri_dms_subj7, fc_fmri_dms_subj8, fc_fmri_dms_subj9,
                        fc_fmri_dms_subj10 ]) 
fc_fmri_ctl = np.array([fc_fmri_ctl_subj1, fc_fmri_ctl_subj2, fc_fmri_ctl_subj3,
                        fc_fmri_ctl_subj4, fc_fmri_ctl_subj5, fc_fmri_ctl_subj6,
                        fc_fmri_ctl_subj7, fc_fmri_ctl_subj8, fc_fmri_ctl_subj9,
                        fc_fmri_ctl_subj10 ]) 

# calculate the mean of correlation coefficients across all given subjects
fc_syn_dms_mean = np.mean(fc_syn_dms, axis=0)
fc_syn_ctl_mean = np.mean(fc_syn_ctl, axis=0)
fc_fmri_dms_mean = np.mean(fc_fmri_dms, axis=0)
fc_fmri_ctl_mean = np.mean(fc_fmri_ctl, axis=0)

# calculate the standard deviation of correlation coefficients across subjects
fc_syn_dms_std = np.std(fc_syn_dms, axis=0)
fc_syn_ctl_std = np.std(fc_syn_ctl, axis=0)
fc_fmri_dms_std = np.std(fc_fmri_dms, axis=0)
fc_fmri_ctl_std = np.std(fc_fmri_ctl, axis=0)

# convert to Pandas dataframe, using the transpose to convert to a format where the names
# of the modules are the labels for each time-series
fc_mean = pd.DataFrame(np.array([fc_syn_dms_mean, fc_syn_ctl_mean,
                                 fc_fmri_dms_mean, fc_fmri_ctl_mean]),
                      columns=np.array(['V1', 'V4', 'D1', 'D2', 'FS', 'FR']),
                       index=np.array(['DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri']))
fc_std  = pd.DataFrame(np.array([fc_syn_dms_std, fc_syn_ctl_std,
                                 fc_fmri_dms_std, fc_fmri_ctl_std]),
                      columns=np.array(['V1', 'V4', 'D1', 'D2', 'FS', 'FR']),
                       index=np.array(['DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri']))

# now, plot means and std's using 'pandas framework...
fig, ax = plt.subplots()
fc_mean.plot(yerr=fc_std, ax=ax, kind='bar')

plt.tight_layout()

# Show the plots on the screen
plt.show()
