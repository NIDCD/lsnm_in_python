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
#   This file (avg_func_conn_across_subjects_tvb.py) was created on September 16, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 16 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/
#
# avg_func_conn_across_subjects_tvb.py
#
# Reads the correlation coefficients from several python (*.npy) data files, each
# corresponding to a single subject, and calculates the average functional connectivity
# across all subjects, as well as the standard deviation for each data point. It uses
# previously calculated correlation coefficients between IT and all other areas in both,
# synapctic activity time-series, and fMRI bold time-series.
#
# The input is synaptic and fmri activity of resting state simulations in TVB
# using the Hagmann's brain

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import pandas as pd

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# construct array of indices of modules contained in an LSNM model, minus 1
modules = np.arange(7)

# construct array of subjects to be considered
subjects = np.arange(1)

# define the names of the output files where the correlation coefficients were stored
func_conn_syn_dms_subj1 = '../visual_model/subject_tvb/output.36trials/corr_syn_IT_vs_all_tvb.npy'
func_conn_fmri_dms_subj1 = '../visual_model/subject_tvb/output.36trials/corr_fmri_IT_vs_all_poisson_tvb.npy'

# open files that contain correlation coefficients
fc_syn_dms_subj1 = np.load(func_conn_syn_dms_subj1)
fc_fmri_dms_subj1 = np.load(func_conn_fmri_dms_subj1)

# convert to Pandas dataframe, using the transpose to convert to a format where the names
# of the modules are the labels for each time-series
fc = pd.DataFrame(np.array([fc_syn_dms_subj1, fc_fmri_dms_subj1]),
                  columns=np.array(['V1', 'V4', 'FS', 'D1', 'D2', 'FR', 'LIT']),
                  index=np.array(['DMS-syn', 'DMS-fmri']))

# now, plot means and std's using 'pandas framework...
fig, ax = plt.subplots()
fc.plot(ax=ax, kind='bar', ylim=[-0.4,1])

plt.tight_layout()

# Show the plots on the screen
plt.show()
