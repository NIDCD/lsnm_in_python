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
#
# avg_func_conn_across_subjects.py
#
# Reads the correlation coefficients from several python (*.npy) data files, each
# corresponding to a single subject, and calculates:
# (1) the average functional connectivity across all subjects,
# (2) means, standard deviations and variances for each data point.
# Means, standard deviations, and variances are stored in a text output file.
# For the calculations, It uses
# previously calculated correlation coefficients between IT and all other areas in both,
# synaptic activity time-series and fMRI bold time-series.
# It also performs a t-test for the comparison between the mean of each condition
# (DMS vs CTL), and displays the t values.

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

# define output file where means, standard deviations, and variances will be stored
fc_stats_FILE = 'fc_stats.txt'

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
func_conn_fmri_dms_subj1 = '../visual_model/subject_1/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj2 = '../visual_model/subject_2/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj3 = '../visual_model/subject_3/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj4 = '../visual_model/subject_4/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj5 = '../visual_model/subject_5/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj6 = '../visual_model/subject_6/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj7 = '../visual_model/subject_7/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj8 = '../visual_model/subject_8/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj9 = '../visual_model/subject_9/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj10 = '../visual_model/subject_10/output.36trials/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_ctl_subj1 = '../visual_model/subject_1/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj2 = '../visual_model/subject_2/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj3 = '../visual_model/subject_3/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj4 = '../visual_model/subject_4/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj5 = '../visual_model/subject_5/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj6 = '../visual_model/subject_6/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj7 = '../visual_model/subject_7/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj8 = '../visual_model/subject_8/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj9 = '../visual_model/subject_9/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj10 = '../visual_model/subject_10/output.36trials/corr_fmri_IT_vs_all_ctl_balloon.npy'

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

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
fc_syn_dms  = np.arctanh(fc_syn_dms)
fc_syn_ctl  = np.arctanh(fc_syn_ctl)
fc_fmri_dms = np.arctanh(fc_fmri_dms)
fc_fmri_ctl = np.arctanh(fc_fmri_ctl)

# increase font size
plt.rcParams.update({'font.size': 30})

plt.figure()
ax1=plt.subplot(2,1,1)
plt.boxplot(fc_syn_dms, showmeans=True)
ax1.set_ylim([-.25, 1.5])
#ax2=plt.subplot(2,2,2)
#plt.boxplot(fc_syn_ctl, showmeans=True)
#ax2.set_ylim([-.25, 1.6])
ax3=plt.subplot(2,1,2)
plt.boxplot(fc_fmri_dms, showmeans=True)
ax3.set_ylim([-.25, 1.5])
#ax4=plt.subplot(2,2,4)
#plt.boxplot(fc_fmri_ctl, showmeans=True)
#ax4.set_ylim([-.25, 1.6])

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

# calculate the variance of the correlation coefficients across subjects
fc_syn_dms_var = np.var(fc_syn_dms, axis=0)
fc_syn_ctl_var = np.var(fc_syn_ctl, axis=0)
fc_fmri_dms_var= np.var(fc_fmri_dms, axis=0)
fc_fmri_ctl_var= np.var(fc_fmri_ctl, axis=0)

# now, convert back to from Z to R correlation coefficients, prior to plotting
fc_syn_dms_mean  = np.tanh(fc_syn_dms_mean)
fc_syn_ctl_mean  = np.tanh(fc_syn_ctl_mean)
fc_fmri_dms_mean = np.tanh(fc_fmri_dms_mean)
fc_fmri_ctl_mean = np.tanh(fc_fmri_ctl_mean)

fc_syn_dms_std  = np.tanh(fc_syn_dms_std)
fc_syn_ctl_std  = np.tanh(fc_syn_ctl_std)
fc_fmri_dms_std = np.tanh(fc_fmri_dms_std)
fc_fmri_ctl_std = np.tanh(fc_fmri_ctl_std)

fc_syn_dms_var  = np.tanh(fc_syn_dms_var)
fc_syn_ctl_var  = np.tanh(fc_syn_ctl_var)
fc_fmri_dms_var = np.tanh(fc_fmri_dms_var)
fc_fmri_ctl_var = np.tanh(fc_fmri_ctl_var)

# ... and save above values to output file
np.savetxt(fc_stats_FILE, [np.append(fc_syn_dms_mean, [fc_syn_ctl_mean,
                                     fc_fmri_dms_mean, fc_fmri_ctl_mean]),
                           np.append(fc_syn_dms_std, [fc_syn_ctl_std,
                                     fc_fmri_dms_std, fc_fmri_ctl_std]),
                           np.append(fc_syn_dms_var, [fc_syn_ctl_var,
                                     fc_fmri_dms_var, fc_fmri_ctl_var] )],
           fmt='%.4f',
           header='Synaptic activities correlation stats (DMS and CTL) grouped by module')

# Calculate the statistical significance by using a one-tailed t-test:
# We are going to have two groups: DMS group and control group (each sample size is 10 subjects)
# Our research hypothesis is:
#          * The correlations in the DMS group are larger than the correlations in the CTL group.
# The NULL hypothesis is:
#          * Correlations in the DMS group are not larger than Correlations in the CTL group.
# The value of alpha (p-threshold) will be 0.05
# STEPS:
#     (1) subtract the mean of control group from the mean of DMS group:
fc_syn_mean_diff = fc_syn_ctl_mean - fc_syn_dms_mean
fc_fmri_mean_diff= fc_fmri_ctl_mean- fc_fmri_dms_mean
#     (2) Calculate, for both control and DMS, the variance divided by sample size minus 1:
fc_syn_ctl_a = fc_syn_ctl_var / 9.0
fc_syn_dms_a = fc_syn_dms_var / 9.0
fc_fmri_ctl_a= fc_fmri_ctl_var / 9.0
fc_fmri_dms_a= fc_fmri_dms_var / 9.0
#     (3) Add results obtained for CTL and DMS in step (2) together:
fc_syn_a = fc_syn_ctl_a + fc_syn_dms_a
fc_fmri_a= fc_fmri_ctl_a+ fc_fmri_dms_a
#     (4) Take the square root the results in step (3):
sqrt_fc_syn_a = np.sqrt(fc_syn_a)
sqrt_fc_fmri_a= np.sqrt(fc_fmri_a)
#     (5) Divide the results of step (1) by the results of step (4) to obtain 't':
fc_syn_t = fc_syn_mean_diff  / sqrt_fc_syn_a
fc_fmri_t= fc_fmri_mean_diff / sqrt_fc_fmri_a
#     (6) Calculate the degrees of freedom (add up number of observations for each group
#         minus number of groups):
dof = 10 + 10 - 2
#     (7) find the p-values for the above 't' and 'degrees of freedom':
fc_syn_p_values  = t.sf(fc_syn_t, dof)
fc_fmri_p_values = t.sf(fc_fmri_t, dof)

print 't-values for synaptic activity correlations: ', fc_syn_t
print 't-values for fmri time-series correlations: ', fc_fmri_t

# convert to Pandas dataframe, using the transpose to convert to a format where the names
# of the modules are the labels for each time-series
fc_mean = pd.DataFrame(np.array([fc_syn_dms_mean, fc_syn_ctl_mean,
                                 fc_fmri_dms_mean, fc_fmri_ctl_mean]),
                      columns=np.array(['V1', 'V4', 'FS', 'D1', 'D2', 'FR', 'LIT']),
                       index=np.array(['DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri']))
#fc_std  = pd.DataFrame(np.array([fc_syn_dms_std, fc_syn_ctl_std,
#                                 fc_fmri_dms_std, fc_fmri_ctl_std]),
#                      columns=np.array(['V1', 'V4', 'D1', 'D2', 'FS', 'FR']),
#                       index=np.array(['DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri']))

# now, plot means and std's using 'pandas framework...

mpl_fig = plt.figure()  # start a new figure

# create more space to the right of the plot for the legend
#ax = mpl_fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax = plt.gca()          # get hold of the axes

bars=fc_mean.plot(ax=ax, kind='bar',
                  color=['yellow', 'green', 'orange', 'red', 'pink', 'purple', 'grey'],
                  ylim=[-0.16,1])

# change the location of the legend
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=7, mode="expand", borderaxespad=0.,prop={'size':30})

ax.set_xticklabels( ('DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri'), rotation=0, ha='center')

#plt.tight_layout()

# Show the plots on the screen
plt.show()

# send figure to plot.ly website for showing others:
#plotly_fig = tls.mpl_to_plotly(mpl_fig)

#unique_url = py.plot(plotly_fig)
