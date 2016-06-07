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
# It also performs a one-tailed paired t-test for the comparison between the mean of each condition
# (DMS vs CTL), and displays the t values.

import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.tools as tls

import matplotlib as mpl

import pandas as pd

from scipy.stats import t

from scipy import stats

import math as m

# set matplot lib parameters to produce visually appealing plots
#mpl.style.use('ggplot')

# construct array of indices of modules contained in an LSNM model, minus 1
modules = np.arange(7)

# construct array of subjects to be considered
subjects = np.arange(10)

# define output file where means, standard deviations, and variances will be stored
fc_stats_FILE = 'fc_stats_with_feedback.txt'

# define the names of the input files where the correlation coefficients were stored
func_conn_syn_dms_subj1 = 'subject_11/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj2 = 'subject_12/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj3 = 'subject_13/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj4 = 'subject_14/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj5 = 'subject_15/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj6 = 'subject_16/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj7 = 'subject_17/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj8 = 'subject_18/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj9 = 'subject_19/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_dms_subj10 = 'subject_20/output.36trials.with_feedback/corr_syn_IT_vs_all_dms.npy'
func_conn_syn_ctl_subj1 = 'subject_11/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj2 = 'subject_12/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj3 = 'subject_13/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj4 = 'subject_14/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj5 = 'subject_15/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj6 = 'subject_16/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj7 = 'subject_17/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj8 = 'subject_18/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj9 = 'subject_19/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_syn_ctl_subj10 = 'subject_20/output.36trials.with_feedback/corr_syn_IT_vs_all_ctl.npy'
func_conn_fmri_dms_subj1 = 'subject_11/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj2 = 'subject_12/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj3 = 'subject_13/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj4 = 'subject_14/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj5 = 'subject_15/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj6 = 'subject_16/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj7 = 'subject_17/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj8 = 'subject_18/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj9 = 'subject_19/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_dms_subj10 = 'subject_20/output.36trials.with_feedback/corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_ctl_subj1 = 'subject_11/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj2 = 'subject_12/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj3 = 'subject_13/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj4 = 'subject_14/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj5 = 'subject_15/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj6 = 'subject_16/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj7 = 'subject_17/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj8 = 'subject_18/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj9 = 'subject_19/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'
func_conn_fmri_ctl_subj10 = 'subject_20/output.36trials.with_feedback/corr_fmri_IT_vs_all_ctl_balloon.npy'

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
plt.rcParams.update({'font.size': 15})

# optional caption for figure
txt = '''Figure 2. Functional connectivities between IT and all other brain modules expressed as the average correlation coefficients among 
the selected ROIs across subjects.  Shown are both task (DMS) and passive viewing control (CTL) conditions and integrated synaptic 
(SYN) and BOLD (fmri) time-series.  The Y-axis represents the correlation coefficient r-value. cIT corresponds to the IT module 
contralateral to the task-based IT module, which generates only noise activity. '''

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

# calculate the standard error of the mean of correlation coefficients across subjects
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
#fc_syn_dms_mean  = np.tanh(fc_syn_dms_mean)
#fc_syn_ctl_mean  = np.tanh(fc_syn_ctl_mean)
#fc_fmri_dms_mean = np.tanh(fc_fmri_dms_mean)
#fc_fmri_ctl_mean = np.tanh(fc_fmri_ctl_mean)

#fc_syn_dms_std  = np.tanh(fc_syn_dms_std)
#fc_syn_ctl_std  = np.tanh(fc_syn_ctl_std)
#fc_fmri_dms_std = np.tanh(fc_fmri_dms_std)
#fc_fmri_ctl_std = np.tanh(fc_fmri_ctl_std)

#fc_syn_dms_var  = np.tanh(fc_syn_dms_var)
#fc_syn_ctl_var  = np.tanh(fc_syn_ctl_var)
#fc_fmri_dms_var = np.tanh(fc_fmri_dms_var)
#fc_fmri_ctl_var = np.tanh(fc_fmri_ctl_var)

# ... and save above values to output file
np.savetxt(fc_stats_FILE, [np.append(fc_syn_dms_mean, [fc_syn_ctl_mean,
                                     fc_fmri_dms_mean, fc_fmri_ctl_mean]),
                           np.append(fc_syn_dms_std, [fc_syn_ctl_std,
                                     fc_fmri_dms_std, fc_fmri_ctl_std]),
                           np.append(fc_syn_dms_var, [fc_syn_ctl_var,
                                     fc_fmri_dms_var, fc_fmri_ctl_var] )],
           fmt='%.4f',
           header='Synaptic and BOLD correlation stats (DMS and CTL) grouped by module')

# Calculate the statistical significance by using a one-tailed paired t-test:
# We are going to have one group of 10 subjects, doing both DMS and control task
# STEPS:
#     (1) Set up hypotheses:
# The NULL hypothesis is:
#          * The mean difference between paired observations (DMS and CTL) is zero
# Our alternative hypothesis is:
#          * The mean difference between paired observations (DMS and CTL) is not zero
#     (2) Set a significance level:
alpha = 0.05
#     (3) What is the critical value and the rejection region?
n = 10 - 1                   # sample size minus 1
rejection_region = 1.833       # as found on t-test table for t and dof given,
                               # values of t above rejection_region will be rejected
#     (4) compute the value of the test statistic                               
# calculate differences between the pairs of data:
d_syn  = fc_syn_dms - fc_syn_ctl
d_fmri = fc_fmri_dms- fc_fmri_ctl
# calculate the mean of those differences
d_syn_mean = np.mean(d_syn, axis=0)
d_fmri_mean= np.mean(d_fmri, axis=0)
# calculate the standard deviation of those differences
d_syn_std = np.std(d_syn, axis=0)
d_fmri_std = np.std(d_fmri, axis=0)
# calculate square root of sample size
sqrt_n = m.sqrt(n)
# calculate standard error of the mean differences
d_syn_sem = d_syn_std/sqrt_n 
d_fmri_sem= d_fmri_std/sqrt_n
# calculate the t statistic:
t_star_syn = d_syn_mean / d_syn_sem
t_star_fmri= d_fmri_mean/ d_fmri_sem

print 't-values for synaptic activity correlations: ', t_star_syn
print 't-values for fmri time-series correlations: ', t_star_fmri

print 'ISA  Mean Differences (IT-V1, IT-V4, IT-FS, IT-D1, IT-D2, IT-FR): ', d_syn_mean  
print 'fMRI Mean Differences (IT-V1, IT-V4, IT-FS, IT-D1, IT-D2, IT-FR): ', d_fmri_mean

print 'Dimensions of mean differences array', d_syn_mean.shape
print 'Dimensions of std of differences array', d_syn_std.shape

# convert to Pandas dataframe, using the transpose to convert to a format where the names
# of the modules are the labels for each time-series
d_mean = pd.DataFrame(np.array([d_syn_mean[:-1],
                                 d_fmri_mean[:-1]]),
                      columns=np.array(['V1',
                                        'V4',
                                        'FS',
                                        'D1',
                                        'D2',
                                        'FR']), #, 'cIT']),
                       index=np.array(['ISA', 'fMRI']))
d_std  = pd.DataFrame(np.array([d_syn_sem[:-1],
                                 d_fmri_sem[:-1]]),
                      columns=np.array(['V1',
                                        'V4',
                                        'FS',
                                        'D1',
                                        'D2',
                                        'FR']), #, 'cIT']),
                       index=np.array(['ISA', 'fMRI']))

# now, plot means and std's using 'pandas framework...

mpl_fig = plt.figure()  # start a new figure

# create more space to the right of the plot for the legend
#ax = mpl_fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax = plt.gca()          # get hold of the axes

bars=d_mean.plot(yerr=d_std, ax=ax, kind='bar',
                  color=['yellow', 'green', 'orange', 'red', 'pink', 'purple'], #, 'lightblue'],
                  ylim=[-0.5, 0.4])

# change the location of the legend
#ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#           ncol=7, mode="expand", borderaxespad=0.,prop={'size':30})

ax.set_xticklabels( ('ISA', 'fMRI'), rotation=0, ha='center')

ax.set_ylabel('Functional connectivity differences')

ax.grid(b=False)
ax.yaxis.grid(True)

#line_fig = plt.figure()

#ax = plt.gca()

#lines=

#plt.tight_layout()

# optional figure caption
#mpl_fig.subplots_adjust(bottom=0.2)
#mpl_fig.text(.1, 0.03, txt)

# Show the plots on the screen
plt.show()

# send figure to plot.ly website for showing others:
#plotly_fig = tls.mpl_to_plotly(mpl_fig)

#unique_url = py.plot(plotly_fig)
