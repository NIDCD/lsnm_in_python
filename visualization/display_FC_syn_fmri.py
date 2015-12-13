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
#   This file (display_FC_syn_fmri.py) was created on November 30, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on November 30 2015
#
# **************************************************************************/
#
# display_FC_syn_fmri.py
#
# Reads the correlation coefficients from 2 (*.npy) data files, 
# corresponding to a mean or representative subject, of synaptc and fmri time-series.
# It displays a bar graph to show those correlations visually.

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

# define the names of the input files where the correlation coefficients were stored
func_conn_syn_dms_subj1 = 'corr_syn_IT_vs_all_dms.npy'
func_conn_syn_ctl_subj1 = 'corr_syn_IT_vs_all_ctl.npy'
func_conn_fmri_dms_subj1 = 'corr_fmri_IT_vs_all_dms_balloon.npy'
func_conn_fmri_ctl_subj1 = 'corr_fmri_IT_vs_all_ctl_balloon.npy'

# open files that contain correlation coefficients
fc_syn_dms = np.load(func_conn_syn_dms_subj1)
fc_syn_ctl = np.load(func_conn_syn_ctl_subj1)
fc_fmri_dms = np.load(func_conn_fmri_dms_subj1)
fc_fmri_ctl = np.load(func_conn_fmri_ctl_subj1)

# increase font size
plt.rcParams.update({'font.size': 30})

# convert to Pandas dataframe, using the transpose to convert to a format where the names
# of the modules are the labels for each time-series
fc_mean = pd.DataFrame(np.array([fc_syn_dms, fc_syn_ctl,
                                 fc_fmri_dms, fc_fmri_ctl]),
                      columns=np.array(['V1', 'V4', 'FS', 'D1', 'D2', 'FR', 'cIT']),
                       index=np.array(['DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri']))
#fc_std  = pd.DataFrame(np.array([fc_syn_dms_std, fc_syn_ctl_std,
#                                 fc_fmri_dms_std, fc_fmri_ctl_std]),
#                      columns=np.array(['V1', 'V4', 'D1', 'D2', 'FS', 'FR']),
#                       index=np.array(['DMS-syn', 'CTL-syn', 'DMS-fmri', 'CTL-fmri']))

# now, plot means using 'pandas framework...

mpl_fig = plt.figure()  # start a new figure

# create more space to the right of the plot for the legend
#ax = mpl_fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax = plt.gca()          # get hold of the axes

bars=fc_mean.plot(ax=ax, kind='bar',
                  color=['yellow', 'green', 'orange', 'red', 'pink', 'purple', 'lightblue'],
                  ylim=[-0.02,1])

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
