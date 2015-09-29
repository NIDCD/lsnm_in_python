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
#   This file (avg_func_conn_across_subjects_boxplot.py) was created on
#   September 20, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 20 2015
#
# **************************************************************************/
#
# avg_func_conn_across_subjects.py
#
# Reads the correlation coefficients from several python (*.npy) data files, each
# corresponding to a single subject, and calculates:
# (1) the average functional connectivity across all subjects, including TVB subject
# (2) means, standard deviations and variances for each data point.
# For the calculations, It uses
# previously calculated correlation coefficients between IT and all other areas in both,
# synaptic activity time-series and fMRI bold time-series.

import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.tools as tls

import matplotlib as mpl

import pandas as pd

from scipy.stats import t

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# construct array of indices of modules contained in an LSNM model, minus 1
modules = np.arange(7)

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
func_conn_syn_dms_subj10= '../visual_model/subject_10/output.36trials/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_tvb   = '../visual_model/subject_tvb/output.36trials/corr_syn_IT_vs_all_tvb.npy'

func_conn_fmri_dms_subj1 = '../visual_model/subject_1/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj2 = '../visual_model/subject_2/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj3 = '../visual_model/subject_3/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj4 = '../visual_model/subject_4/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj5 = '../visual_model/subject_5/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj6 = '../visual_model/subject_6/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj7 = '../visual_model/subject_7/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj8 = '../visual_model/subject_8/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj9 = '../visual_model/subject_9/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_subj10= '../visual_model/subject_10/output.36trials/corr_fmri_IT_vs_all_dms_poisson.npy'
func_conn_fmri_dms_tvb   = '../visual_model/subject_tvb/output.36trials/corr_fmri_IT_vs_all_poisson_tvb.npy'

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
fc_syn_dms_tvb = np.load(func_conn_syn_dms_tvb)
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
fc_fmri_dms_tvb = np.load(func_conn_fmri_dms_tvb)

# construct numpy arrays that contain correlation coefficients for all subjects
# (the functional connectivity of IT versus the other 6 modules of LSNM model and contralateral IT)
fc_syn_dms = np.array([fc_syn_dms_subj1, fc_syn_dms_subj2, fc_syn_dms_subj3,
                       fc_syn_dms_subj4, fc_syn_dms_subj5, fc_syn_dms_subj6,
                       fc_syn_dms_subj7, fc_syn_dms_subj8, fc_syn_dms_subj9,
                       fc_syn_dms_subj10, fc_syn_dms_tvb ]) 
fc_fmri_dms = np.array([fc_fmri_dms_subj1, fc_fmri_dms_subj2, fc_fmri_dms_subj3,
                        fc_fmri_dms_subj4, fc_fmri_dms_subj5, fc_fmri_dms_subj6,
                        fc_fmri_dms_subj7, fc_fmri_dms_subj8, fc_fmri_dms_subj9,
                        fc_fmri_dms_subj10, fc_fmri_dms_tvb ]) 

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
fc_syn_dms  = np.arctanh(fc_syn_dms)
fc_fmri_dms = np.arctanh(fc_fmri_dms)

# concatenate both datasets together prior to generating boxplot
fc_dms = np.concatenate((fc_syn_dms, fc_fmri_dms), axis=1)

plt.figure()
ax1=plt.subplot()
bp=plt.boxplot(fc_dms)
ax1.set_ylim([-.5, 1.5])
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

plt.tight_layout()

# Show the plots on the screen
plt.show()

# send figure to plot.ly website for showing others:
#plotly_fig = tls.mpl_to_plotly(mpl_fig)

#unique_url = py.plot(plotly_fig)
