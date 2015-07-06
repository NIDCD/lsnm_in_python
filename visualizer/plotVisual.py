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
#   This file (plotVisual.py) was created on December 1, 2014.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 13 2015  
# **************************************************************************/

# plotVisual.py
#
# Plot output data files of visual delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# Load data files
lgns = np.loadtxt('../visual_model/output/lgns.out')
efd1 = np.loadtxt('../visual_model/output/efd1.out')
efd2 = np.loadtxt('../visual_model/output/efd2.out')
ev1h = np.loadtxt('../visual_model/output/ev1h.out')
ev1v = np.loadtxt('../visual_model/output/ev1v.out')
ev4c = np.loadtxt('../visual_model/output/ev4c.out')
ev4h = np.loadtxt('../visual_model/output/ev4h.out')
ev4v = np.loadtxt('../visual_model/output/ev4v.out')
exfr = np.loadtxt('../visual_model/output/exfr.out')
exfs = np.loadtxt('../visual_model/output/exfs.out')
exss = np.loadtxt('../visual_model/output/exss.out')

tvb_v1 = np.loadtxt('../visual_model/output/ev1h_tvb.out')
tvb_v4 = np.loadtxt('../visual_model/output/ev4h_tvb.out')
tvb_it = np.loadtxt('../visual_model/output/exss_tvb.out')
tvb_fs = np.loadtxt('../visual_model/output/exfs_tvb.out')
tvb_d1 = np.loadtxt('../visual_model/output/efd1_tvb.out')
tvb_d2 = np.loadtxt('../visual_model/output/efd2_tvb.out')
tvb_fr = np.loadtxt('../visual_model/output/exfr_tvb.out')

# Extract number of timesteps from one of the matrices
timesteps = lgns.shape[0]

print lgns.shape[0]

# Contruct a numpy array of timesteps (data points provided in data file)
t = np.arange(0, timesteps, 1)

# Set up plot
fig1=plt.figure(1, facecolor='white')

# Plot V1 module
ax = plt.subplot(5,1,1)
ax.plot(t, tvb_v1, color='b', linewidth=2)
ax.plot(t, ev1h, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
ax.set_title('SIMULATED ELECTRICAL ACTIVITY, V1 and V4')
plt.ylabel('V1h', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(5,1,2)
ax.plot(t, tvb_v1, color='b', linewidth=2)
ax.plot(t, ev1v, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('V1v', rotation='horizontal', horizontalalignment='right')

# Plot V4 module
ax = plt.subplot(5,1,3)
ax.plot(t, tvb_v4, color='b', linewidth=2)
ax.plot(t, ev4h, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('V4h', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(5,1,4)
ax.plot(t, tvb_v4, color='b', linewidth=2)
ax.plot(t, ev4c, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('V4c', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(5,1,5)
l1 = ax.plot(t, tvb_v4, color='b', linewidth=2)
l2 = ax.plot(t, ev4v, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('V4v', rotation='horizontal', horizontalalignment='right')

ax.set_xlabel('Non-specific')

plt.tight_layout()

ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)

# start a second figure 
plt.figure(2, facecolor='white')

# Plot IT module
ax = plt.subplot(5,1,1)
ax.plot(t, tvb_it, color='b', linewidth=2)
ax.plot(t, exss, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_title('SIMULATED ELECTRICAL ACTIVITY, IT and PFC')
ax.set_xlim(0,timesteps)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')

# Plot PFC modules FS, FD1, and FD2
ax = plt.subplot(5,1,2)
ax.plot(t, tvb_fs, color='b', linewidth=2)
ax.plot(t, exfs, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(5,1,3)
ax.plot(t, tvb_d1, color='b', linewidth=2)
ax.plot(t, efd1, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(5,1,4)
ax.plot(t, tvb_d2, color='b', linewidth=2)
ax.plot(t, efd2, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

# Plot FR (Response module)
ax = plt.subplot(5,1,5)
ax.plot(t, tvb_fr, color='b', linewidth=2)
ax.plot(t, exfr, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('R', rotation='horizontal', horizontalalignment='right')

plt.tight_layout()

# Show the plot on the screen
plt.show()

