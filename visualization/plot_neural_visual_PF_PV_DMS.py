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
#   This file (plot_neural_visual_PF_PV_DMS.py) was created on December 10 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on December 11 2017 
# **************************************************************************/

# plot_neural_visual_PF_PV_DMS.py
#
# Plot output data files of visual delay-match-to-sample simulation
# during three conditions: PF, PV, and DMS

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# Load data files
ev1h_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/ev1h.out')
ev4h_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/ev4h.out')
exss_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/exss.out')
exfs_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/exfs.out')
efd1_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/efd1.out')
efd2_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/efd2.out')
exfr_PF = np.loadtxt('subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/exfr.out')

ev1h_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/ev1h.out')
ev4h_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/ev4h.out')
exss_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/exss.out')
exfs_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/exfs.out')
efd1_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/efd1.out')
efd2_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/efd2.out')
exfr_PV = np.loadtxt('subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/exfr.out')

ev1h_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/ev1h.out')
ev4h_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/ev4h.out')
exss_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/exss.out')
exfs_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/exfs.out')
efd1_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/efd1.out')
efd2_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/efd2.out')
exfr_DMS = np.loadtxt('subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/exfr.out')

# Extract number of timesteps from one of the matrices
timesteps = ev1h_PF.shape[0]

print timesteps

# the following variable defines the timesteps we will see in the resulting plot
# we also convert the number of timesteps to seconds by multiplying by 50 and dividng by 1000
#start = 3200
#end = 3300
start = 3750
end = 3850
ts_to_plot = end - start
x_lim = ts_to_plot * 50. / 1000.

# Construct a numpy array of timesteps (data points provided in data file)
t = np.linspace(0, (ts_to_plot-1)*50./1000., num=ts_to_plot)

# Set up plot
fig1=plt.figure(1, facecolor='white')

# set up title
plt.suptitle('Average neural activity of task-related brain regions')

# increase font size
plt.rcParams.update({'font.size': 15})

# Plot V1 module
ax = plt.subplot(7,1,1)
#try:
#    ax.plot(t, tvb_v1[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, ev1h_PF[start:end, 15], color='blue', linewidth=3)
#ax.plot(t, ev1h_PV[start:end, 15], color='green', linewidth=3)
#ax.plot(t, ev1h_DMS[start:end,15], color='red', linewidth=3)
ax.plot(t, np.mean(ev1h_PF[start:end], axis=1), color='blue', linewidth=3)
ax.plot(t, np.mean(ev1h_PV[start:end], axis=1), color='green', linewidth=3)
ax.plot(t, np.mean(ev1h_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.setp(ax.get_xticklabels(), visible=False)
#ax.set_title('SIMULATED ELECTRICAL ACTIVITY, V1 and V4')
plt.ylabel('V1', rotation='horizontal', horizontalalignment='right')

# Plot V4 module
ax = plt.subplot(7,1,2)
#try:
#    ax.plot(t, tvb_v4[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, ev4h_PF[start:end, 35], color='blue', linewidth=3)
#ax.plot(t, ev4h_PV[start:end, 35], color='green', linewidth=3)
#ax.plot(t, ev4h_DMS[start:end, 35], color='red', linewidth=3)
ax.plot(t, np.mean(ev4h_PF[start:end], axis=1), color='blue', linewidth=3)
ax.plot(t, np.mean(ev4h_PV[start:end], axis=1), color='green', linewidth=3)
ax.plot(t, np.mean(ev4h_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('V4', rotation='horizontal', horizontalalignment='right')

# Plot IT module
ax = plt.subplot(7,1,3)
#try:
#    ax.plot(t, tvb_it[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, exss_PF[start:end, 5], color='blue', linewidth=3)
#ax.plot(t, exss_PV[start:end, 5], color='green', linewidth=3)
#ax.plot(t, exss_DMS[start:end, 5], color='red', linewidth=3)
ax.plot(t, np.mean(exss_PF[start:end], axis=1), color='blue', linewidth=3)
ax.plot(t, np.mean(exss_PV[start:end], axis=1), color='green', linewidth=3)
ax.plot(t, np.mean(exss_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
#ax.set_title('SIMULATED ELECTRICAL ACTIVITY, IT and PFC')
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')

# Plot PFC modules FS, FD1, and FD2
ax = plt.subplot(7,1,4)
#try:
#    ax.plot(t, tvb_fs[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, exfs_PF[start:end, 5], color='blue', linewidth=3)
#ax.plot(t, exfs_PV[start:end, 5], color='green', linewidth=3)
#ax.plot(t, exfs_DMS[start:end, 5], color='red', linewidth=3)
ax.plot(t, np.mean(exfs_PF[start:end], axis=1), color='blue', linewidth=3)
ax.plot(t, np.mean(exfs_PV[start:end], axis=1), color='green', linewidth=3)
ax.plot(t, np.mean(exfs_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,5)
#try:
#    ax.plot(t, tvb_d1[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, efd1_PF[start:end, 5], color='blue', linewidth=3)
#ax.plot(t, efd1_PV[start:end, 5], color='green', linewidth=3)
#ax.plot(t, efd1_DMS[start:end, 5], color='red', linewidth=3)
ax.plot(t, np.mean(efd1_PF[start:end], axis=1), color='blue', linewidth=3)
ax.plot(t, np.mean(efd1_PV[start:end], axis=1), color='green', linewidth=3)
ax.plot(t, np.mean(efd1_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,6)
#try:
#    ax.plot(t, tvb_d2[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, efd2_PF[start:end, 5], color='blue', linewidth=3)
#ax.plot(t, efd2_PV[start:end, 5], color='green', linewidth=3)
#ax.plot(t, efd2_DMS[start:end, 5], color='red', linewidth=3)
ax.plot(t, np.mean(efd2_PF[start:end], axis=1), color='blue', linewidth=3)
ax.plot(t, np.mean(efd2_PV[start:end], axis=1), color='green', linewidth=3)
ax.plot(t, np.mean(efd2_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

# Plot FR (Response module)
ax = plt.subplot(7,1,7)
#try:
#    ax.plot(t, tvb_fr[0:ts_to_plot], color='b', linewidth=2)
#except:
#    pass
#ax.plot(t, exfr_PF[start:end, 11], color='blue', linewidth=3)
#ax.plot(t, exfr_PV[start:end, 11], color='green', linewidth=3)
#ax.plot(t, exfr_DMS[start:end, 11], color='red', linewidth=3)
l1, = ax.plot(t, np.mean(exfr_PF[start:end], axis=1), color='blue', linewidth=3)
l2, = ax.plot(t, np.mean(exfr_PV[start:end], axis=1), color='green', linewidth=3)
l3, = ax.plot(t, np.mean(exfr_DMS[start:end], axis=1), color='red', linewidth=3)
ax.set_yticks([])
ax.set_xlim(0,x_lim)
#ax.set_ylim(0.,1.)
plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')

plt.xlabel('Time (s)')

#plt.figlegend((l1, l2, l3), ('PF', 'PV', 'DMS'), bbox_to_anchor=(0, 0, 1, 1), loc=2, borderaxespad=0.)

# Show the plot on the screen
plt.show()

