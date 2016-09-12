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
#   This file (plot_syn_and_BOLD.py) was created on December 6, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 30 2016
#
# **************************************************************************/

# plot_syn_and_BOLD.py
#
# Reads the synaptic and BOLD timeseries from a python (*.npy) data files, and it
# displays, separately, synaptic and BOLD timeseries, per each brain area. It also
# displays grey bands in the plot are to show where the control trials are located
# in the timescale.

import numpy as np
import matplotlib.pyplot as plt

experiment_length = 3960

# scans that were removed from BOLD computation
scans_removed = 0

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# Time for one complete trial in milliseconds
Ttrial = 5.5

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

num_of_scans = T / Tr - scans_removed

num_of_subjects = 10

num_of_modules = 8

# construct array of indices of modules contained in an LSNM model
modules = np.arange(num_of_modules)

# define the name of the input files where the BOLD and synaptic timeseries are
# stored:
syn_subj = 'synaptic_in_ROI.npy'
BOLD_subj = 'lsnm_bold_balloon.npy'
            
# open files that contain synaptic and fMRI BOLD timeseries
lsnm_syn = np.load(syn_subj)
lsnm_BOLD = np.load(BOLD_subj)

# construct array indexing scan number of the BOLD timeseries
# (take into account scans that were removed)
total_scans = lsnm_BOLD[0].shape[-1] + scans_removed
total_time = total_scans * Tr
time_removed = scans_removed * Tr
BOLD_timescale = np.arange(time_removed, total_time, Tr)
print BOLD_timescale

# increase font size prior to plotting
plt.rcParams.update({'font.size': 15})

# optional caption for figure
#txt = '''Figure 1. Simulated fMRI BOLD signal using the Balloon hemodynamic
#response model in combined visual LSNM/TVB modules, corresponding to a
#representative subject.  Thirty-six trials were simulated in groups of six
#task (DMS) trials followed by six passive viewing trials. X-axis represents
#time in seconds and the Y-axis is in arbitrary coordinates.  Grey areas
#highlight the timing of the passive viewing trials. FS, D1, D2 and FR represent
#submodules within the PFC region. '''


# Set up figure to plot synaptic signal
fig1=plt.figure(1)

#plt.suptitle('SYNAPTIC ACTIVITIES')

# plot V1 syn time-series in yellow
ax = plt.subplot(7,1,1)
ax.set_yticks([])
ax.set_xticks([])
ax.plot(lsnm_syn[0,1:], linewidth=3.0, color='yellow')

plt.ylabel('V1/V2', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot V4 syn time-series in green
ax = plt.subplot(7,1,2)
ax.plot(lsnm_syn[1,1:], linewidth=3.0, color='lime')
ax.set_yticks([])
ax.set_xticks([])

plt.ylabel('V4', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot IT syn time-series in blue
ax = plt.subplot(7,1,3)
ax.plot(lsnm_syn[2,1:], linewidth=3.0, color='blue')
ax.set_yticks([])
ax.set_xticks([])

plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot FS syn time-series in orange
ax = plt.subplot(7,1,4)
ax.plot(lsnm_syn[3,1:], linewidth=3.0, color='orange')
ax.set_yticks([])
ax.set_xticks([])

plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot D1 syn time-series in red
ax = plt.subplot(7,1,5)
ax.plot(lsnm_syn[4,1:], linewidth=3.0, color='red')
ax.set_yticks([])
ax.set_xticks([])

plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot D2 syn time-series in pink
ax = plt.subplot(7,1,6)
ax.plot(lsnm_syn[5,1:], linewidth=3.0, color='pink')
ax.set_yticks([])
ax.set_xticks([])

plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot FR syn time-series in purple
ax = plt.subplot(7,1,7)
ax.plot(lsnm_syn[6,1:], linewidth=3.0, color='darkorchid')
ax.set_yticks([])

plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')


#set up figure to plot BOLD signal
fig2=plt.figure(2)


#plt.suptitle('fMRI BOLD SIGNAL')

# plot V1 BOLD time-series in yellow
ax = plt.subplot(7,1,1)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)
ax.plot(BOLD_timescale, lsnm_BOLD[0], linewidth=3.0, color='yellow')

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('V1/V2', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot V4 BOLD time-series in green
ax = plt.subplot(7,1,2)
ax.plot(BOLD_timescale, lsnm_BOLD[1], linewidth=3.0, color='lime')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('V4', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot IT BOLD time-series in blue
ax = plt.subplot(7,1,3)
ax.plot(BOLD_timescale, lsnm_BOLD[2], linewidth=3.0, color='blue')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot FS BOLD time-series in orange
ax = plt.subplot(7,1,4)
ax.plot(BOLD_timescale, lsnm_BOLD[3], linewidth=3.0, color='orange')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot D1 BOLD time-series in red
ax = plt.subplot(7,1,5)
ax.plot(BOLD_timescale, lsnm_BOLD[4], linewidth=3.0, color='red')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot D2 BOLD time-series in pink
ax = plt.subplot(7,1,6)
ax.plot(BOLD_timescale, lsnm_BOLD[5], linewidth=3.0, color='pink')
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,200)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# plot FR BOLD time-series in purple
ax = plt.subplot(7,1,7)
ax.plot(BOLD_timescale, lsnm_BOLD[6], linewidth=3.0, color='darkorchid')
ax.set_yticks([])
ax.set_xlim(0,200)

# display gray bands in figure area to show where control blocks are located
ax.axvspan(17.5, 34.0, facecolor='gray', alpha=0.6)
ax.axvspan(50.5, 67.0, facecolor='gray', alpha=0.6)
ax.axvspan(83.5, 100.0, facecolor='gray', alpha=0.6)
ax.axvspan(116.5, 133.0, facecolor='gray', alpha=0.6)
ax.axvspan(149.5, 166.0, facecolor='gray', alpha=0.6)
ax.axvspan(182.5, 199.0, facecolor='gray', alpha=0.6)
plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')
plt.gca().set_axis_bgcolor('black')

# optional figure caption
#fig2.subplots_adjust(bottom=0.2)
#fig2.text(.1, 0.03, txt)

# Show the plots on the screen
plt.show()
