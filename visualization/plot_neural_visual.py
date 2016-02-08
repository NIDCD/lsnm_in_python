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
#   This file (plot_neural_visual.py) was created on December 1, 2014.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on July 19 2015  
# **************************************************************************/

# plot_neural_visual.py
#
# Plot output data files of visual delay-match-to-sample simulation

# what are the locations of relevant TVB nodes within TVB array?
v1_loc = 345
v4_loc = 393
it_loc = 413
d1_loc =  74
d2_loc =  41
fs_loc =  47
fr_loc = 125

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# Load data files
lgns = np.loadtxt('lgns.out')
efd1 = np.loadtxt('efd1.out')
efd2 = np.loadtxt('efd2.out')
ev1h = np.loadtxt('ev1h.out')
ev1v = np.loadtxt('ev1v.out')
ev4c = np.loadtxt('ev4c.out')
ev4h = np.loadtxt('ev4h.out')
ev4v = np.loadtxt('ev4v.out')
exfr = np.loadtxt('exfr.out')
exfs = np.loadtxt('exfs.out')
exss = np.loadtxt('exss.out')

# load file containing TVB nodes electrical activity
try:
    tvb = np.load('tvb_neuronal.npy')
    
    tvb_v1 = tvb[:, 0, v1_loc, 0]
    tvb_v4 = tvb[:, 0, v4_loc, 0]
    tvb_it = tvb[:, 0, it_loc, 0]
    tvb_fs = tvb[:, 0, fs_loc, 0]
    tvb_d1 = tvb[:, 0, d1_loc, 0]
    tvb_d2 = tvb[:, 0, d2_loc, 0]
    tvb_fr = tvb[:, 0, fr_loc, 0]
except:
    pass

# Extract number of timesteps from one of the matrices
timesteps = lgns.shape[0]

print timesteps

# the following variable defines the timesteps we will see in the resulting plot
# we also convert the number of timesteps to seconds by multiplying by 50 and dividng by 1000
ts_to_plot = 660
x_lim = ts_to_plot * 50. / 1000.

# Construct a numpy array of timesteps (data points provided in data file)
t = np.linspace(0, (ts_to_plot-1)*50./1000., num=ts_to_plot)

# Set up plot
fig1=plt.figure(1, facecolor='white')

# increase font size
plt.rcParams.update({'font.size': 15})

# optional caption for figure
#txt = '''Figure 1. Plots of electrical activity of all submodules of the LSNM visual model (red lines) and host nodes of the TVB connectome 
#(blue lines), for a representative simulated subject. The simulated tasks were: three DMS trials (match, mismatch, match), followed 
#by three control trials where degraded shapes were used as inputs and the attention parameter was set to low (passive viewing). 
#The y axis of each of the plots represents level of electrical activity, between zero and one. The x axis of the plots represents 
#time in seconds.'''

# Plot V1 module
ax = plt.subplot(10,1,1)
try:
    ax.plot(t, tvb_v1[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, ev1h[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
#ax.set_title('SIMULATED ELECTRICAL ACTIVITY, V1 and V4')
plt.ylabel('V1h', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(10,1,2)
try:
    ax.plot(t, tvb_v1[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, ev1v[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('V1v', rotation='horizontal', horizontalalignment='right')

# Plot V4 module
ax = plt.subplot(10,1,3)
try:
    ax.plot(t, tvb_v4[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, ev4h[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('V4h', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(10,1,4)
try:
    ax.plot(t, tvb_v4[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, ev4c[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('V4c', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(10,1,5)
try:
    ax.plot(t, tvb_v4[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, ev4v[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('V4v', rotation='horizontal', horizontalalignment='right')

#plt.tight_layout()

# Plot IT module
ax = plt.subplot(10,1,6)
try:
    ax.plot(t, tvb_it[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, exss[0:ts_to_plot], color='r')
ax.set_yticks([])
#ax.set_title('SIMULATED ELECTRICAL ACTIVITY, IT and PFC')
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')

# Plot PFC modules FS, FD1, and FD2
ax = plt.subplot(10,1,7)
try:
    ax.plot(t, tvb_fs[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, exfs[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(10,1,8)
try:
    ax.plot(t, tvb_d1[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, efd1[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(10,1,9)
try:
    ax.plot(t, tvb_d2[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, efd2[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

# Plot FR (Response module)
ax = plt.subplot(10,1,10)
try:
    ax.plot(t, tvb_fr[0:ts_to_plot], color='b', linewidth=2)
except:
    pass
ax.plot(t, exfr[0:ts_to_plot], color='r')
ax.set_yticks([])
ax.set_xlim(0,x_lim)
plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')

plt.xlabel('Time (s)')

#plt.tight_layout()

# optional figure caption
#fig1.subplots_adjust(bottom=0.2)
#fig1.text(.1, 0.03, txt)

# Show the plot on the screen
plt.show()

