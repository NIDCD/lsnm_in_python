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
#   This file (plot_syn_and_BOLD_tvb.py) was created on August 30, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on August 30, 2016  
# **************************************************************************/

# plot_syn_and_BOLD_tvb.py
#
# Plot synaptic and BOLD files of Hagmann's brain using both TVB and Hybrid TVB/LSNM
# simulations (side to side)

# what are the locations of relevant TVB nodes within TVB array?
# the following are the so-called 'host nodes'
v1_loc = 345
v4_loc = 393
it_loc = 413
fs_loc =  47
d1_loc =  74
d2_loc =  41
fr_loc = 125

# what other TVB nodes are the TVB host nodes connected to (with weights above 0.5)?
# the nodes below were taken from an output given by the script:
# "display_hagmanns_brain_connectivity.py"
v1_cxn = [334, 339, 340, 341, 342, 343, 344, 346, 347, 348, 350, 351, 352, 353,
          363, 373, 374, 375, 376, 842, 848, 874, 877]
v4_cxn = [379, 391, 392, 397, 398, 401, 423]
it_cxn = [380, 401, 402, 412]
fs_cxn = [39, 43, 44, 45, 48, 49, 50, 51, 52, 53, 107]
d1_cxn = [41, 42, 44, 48, 53, 54, 55, 67, 71, 72, 73, 75, 76, 77, 129, 130]
d2_cxn = [1, 19, 20, 21, 22, 39, 40, 42, 43, 44, 45, 53, 54, 64, 65, 66, 73, 74, 75]
fr_cxn = [71, 78, 93, 104, 105, 106, 114, 115, 126, 128, 129, 130, 131, 132, 133]


import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

# load file containing TVB nodes electrical activity
tvb = np.load('output.36trials/tvb_synaptic.npy')
hybrid_tvb_lsnm = np.load('output.36trials.with_feedback/tvb_synaptic.npy')


# the following nodes have the strongest connections with host nodes
# ... as determined by script "display_hagmann_brain_connectivity"
tvb_v1 = tvb[:, 0, 346, 0]
tvb_v4 = tvb[:, 0, 391, 0]
tvb_it = tvb[:, 0, 412, 0]
tvb_fs = tvb[:, 0, 44, 0]
tvb_d1 = tvb[:, 0, 42, 0]
tvb_d2 = tvb[:, 0, 66, 0]
tvb_fr = tvb[:, 0, 105, 0]

hybrid_v1 = hybrid_tvb_lsnm[:, 0, 346, 0]
hybrid_v4 = hybrid_tvb_lsnm[:, 0, 391, 0]
hybrid_it = hybrid_tvb_lsnm[:, 0, 412, 0]
hybrid_fs = hybrid_tvb_lsnm[:, 0, 44, 0]
hybrid_d1 = hybrid_tvb_lsnm[:, 0, 42, 0]
hybrid_d2 = hybrid_tvb_lsnm[:, 0, 66, 0]
hybrid_fr = hybrid_tvb_lsnm[:, 0, 105, 0]

# Extract number of timesteps from one of the matrices
timesteps = tvb_v1.shape[0]

# what was the duration of simulation in real time (in ms)?
real_duration = 198

print timesteps

# Contruct a numpy array of timesteps (data points provided in data file)
real_time = np.linspace(0, real_duration, num=timesteps)

# increase font size
plt.rcParams.update({'font.size': 20})

# Set up plot
plt.figure()

# Plot V1 module
ax = plt.subplot(7,1,7)
ax.plot(real_time, tvb_v1, color='k',  linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_v1, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xlim([0, 5])
#ax.set_ylim([0, 1])
#ax.set_title('SIMULATED ELECTRICAL ACTIVITY, HAGMANNS BRAIN')
plt.ylabel('V1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,6)
ax.plot(real_time, tvb_v4, color='k',  linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_v4, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
#ax.set_ylim([0, 1])
plt.ylabel('V4', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,5)
ax.plot(real_time, tvb_it, color='k',  linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_it, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
#ax.set_ylim([0, 1])
plt.ylabel('IT', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,4)
ax.plot(real_time, tvb_fs, color='k',  linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_fs, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
#ax.set_ylim([0, 1])
plt.ylabel('FS', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,3)
ax.plot(real_time, tvb_d1, color='k', linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_d1, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
#ax.set_ylim([0, 1])
plt.ylabel('D1', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,2)
ax.plot(real_time, tvb_d2, color='k', linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_d2, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
#ax.set_ylim([0, 1])
plt.ylabel('D2', rotation='horizontal', horizontalalignment='right')

ax = plt.subplot(7,1,1)
ax.plot(real_time, tvb_fr, color='k', linestyle='--', linewidth=2)
ax.plot(real_time, hybrid_fr, color='k', linewidth=2)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0, 5)
#ax.set_ylim([0, 1])
plt.ylabel('FR', rotation='horizontal', horizontalalignment='right')

#plt.tight_layout()

# Show the plot on the screen
plt.show()

