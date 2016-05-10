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
#   This file (plot_BOLD_scatter_plots.py) was created on May 9, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on May 9, 2016
#
# **************************************************************************/

# plot_BOLD_scatter_plots.py
#
# Displays scatter plots of BOLD timseries in IT vs all other areas.
#
# Reads an NPY file that containts a numpy array with time-series with
# all of the ROIs incuded.
#
#
# The input data (synaptic activities) and the output (BOLD time-series) are numpy arrays
# with columns in the following order:
#
# V1 ROI (right hemisphere, includes LSNM units and TVB nodes) 
# V4 ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (right hemisphere, includes LSNM units and TVB nodes)
# FS ROI (right hemisphere, includes LSNM units and TVB nodes)
# D1 ROI (right hemisphere, includes LSNM units and TVB nodes)
# D2 ROI (right hemisphere, includes LSNM units and TVB nodes)
# FR ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (left hemisphere, contains only  TVB nodes)

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats.stats import pearsonr

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'tvb_bold_balloon.npy'

# read the input file that contains the synaptic activities of all ROIs
bold = np.load(BOLD_file)

print 'LENGTH OF BOLD TIME-SERIES: ', bold[0].size

v1_bold = bold[0, 8:]
v4_bold = bold[1, 8:]
it_bold = bold[2, 8:]
fs_bold = bold[3, 8:]
d1_bold = bold[4, 8:]
d2_bold = bold[5, 8:]
fr_bold = bold[6, 8:]
lit_bold= bold[7, 8:]


# Extract number of timesteps from one of the synaptic activity arrays
bold_timesteps = v1_bold.size
print 'Size of BOLD fMRI arrays: ', bold_timesteps

print 'CORRELATION IT vs V1/V2: ', pearsonr(v1_bold, it_bold)[0]
print 'CORRELATION IT vs V4: ', pearsonr(v4_bold, it_bold)[0]
print 'CORRELATION IT vs FS: ', pearsonr(fs_bold, it_bold)[0]
print 'CORRELATION IT vs D1: ', pearsonr(d1_bold, it_bold)[0]
print 'CORRELATION IT vs D2: ', pearsonr(d2_bold, it_bold)[0]
print 'CORRELATION IT vs FR: ', pearsonr(fr_bold, it_bold)[0]
print 'CORRELATION IT vs contralateral IT: ', pearsonr(lit_bold, it_bold)[0]
print 'CORRELATION IT vs IT: ', pearsonr(it_bold, it_bold)[0]

# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('FR vs contralateral IT (BOLD fMRI)')
plt.scatter(fr_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('D1 vs contralateral IT (BOLD fMRI)')
plt.scatter(d1_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('D2 vs contralateral IT (BOLD fMRI)')
plt.scatter(d2_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('FS vs contralateral IT (BOLD fMRI)')
plt.scatter(fs_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('V4 vs contralateral IT (BOLD fMRI)')
plt.scatter(v4_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('V1/V2 vs contralateral IT (BOLD fMRI)')
plt.scatter(v1_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('IT vs contralateral IT (BOLD fMRI)')
plt.scatter(lit_bold, it_bold)
# Set up figure to plot scatter plots
plt.figure()
plt.suptitle('IT vs  IT (BOLD fMRI)')
plt.scatter(it_bold, it_bold)

plt.figure()
plt.suptitle('V1 and IT BOLD time-series')
plt.plot(v1_bold)
plt.plot(it_bold)

# Show the plots on the screen
plt.show()
