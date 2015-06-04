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

# Load data files
lgns = np.loadtxt('../../output/lgns.out')
efd1 = np.loadtxt('../../output/efd1.out')
efd2 = np.loadtxt('../../output/efd2.out')
ev1h = np.loadtxt('../../output/ev1h.out')
ev1v = np.loadtxt('../../output/ev1v.out')
ev4c = np.loadtxt('../../output/ev4c.out')
ev4h = np.loadtxt('../../output/ev4h.out')
ev4v = np.loadtxt('../../output/ev4v.out')
exfr = np.loadtxt('../../output/exfr.out')
exfs = np.loadtxt('../../output/exfs.out')
exss = np.loadtxt('../../output/exss.out')

tvb_v1 = np.loadtxt('../../output/ev1h_tvb.out')
tvb_st = np.loadtxt('../../output/exss_tvb.out')
tvb_pf = np.loadtxt('../../output/exfs_tvb.out')

# Extract number of timesteps from one of the matrices
timesteps = lgns.shape[0]

# Contruct a numpy array of timesteps (data points provided in data file)
t = np.arange(0, timesteps, 1)

# Set up plot
plt.figure(1)

plt.suptitle('SIMULATED NEURAL ACTIVITY')

# Plot LGN module
ax = plt.subplot(14,1,1)
ax.plot(t, lgns)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('LGN', rotation='horizontal', horizontalalignment='right')

# Plot V1 module
ax = plt.subplot(14,1,2)
ax.plot(t, ev1h)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('V1h', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(14,1,3)
ax.plot(t, ev1v)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('V1v', rotation='horizontal', horizontalalignment='right')

# Plot V4 module
ax = plt.subplot(14,1,4)
ax.plot(t, ev4h)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('V4h', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(14,1,5)
ax.plot(t, ev4c)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('V4c', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(14,1,6)
ax.plot(t, ev4v)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('V4v', rotation='horizontal', horizontalalignment='right')

# Plot IT module
ax = plt.subplot(14,1,7)
ax.plot(t, exss)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')

# Plot PFC modules FS, FD1, and FD2
ax = plt.subplot(14,1,8)
ax.plot(t, exfs)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(14,1,9)
ax.plot(t, efd1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(14,1,10)
ax.plot(t, efd2)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

# Plot FR (Response module)
ax = plt.subplot(14,1,11)
ax.plot(t, exfr)
ax.set_yticks([])
ax.set_xlim(0,timesteps)
plt.ylabel('R', rotation='horizontal', horizontalalignment='right')

# Plot TVB V1 (host node)
ax = plt.subplot(14,1,12)
ax.plot(tvb_v1)
ax.set_yticks([])
#ax.set_xlim(0, timesteps)
plt.ylabel('TVB_rV1', rotation='horizontal', horizontalalignment='right')

# Plot TVB ST (host node)
ax = plt.subplot(14,1,13)
ax.plot(tvb_st)
ax.set_yticks([])
#ax.set_xlim(0, timesteps)
plt.ylabel('TVB_rST', rotation='horizontal', horizontalalignment='right')

# Plot TVB PFC (host node)
ax = plt.subplot(14,1,14)
ax.plot(tvb_pf)
ax.set_yticks([])
#ax.set_xlim(0, timesteps)
plt.ylabel('TVB_rPF', rotation='horizontal', horizontalalignment='right')


plt.xlabel('Timesteps (i.e., Data points)')

# Show the plot on the screen
plt.show()

