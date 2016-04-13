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
#   This file (compute_NAI_auditory.py) was created on March 22, 2016.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on March 22, 2016  
# **************************************************************************/

# compute_NAI_auditory.py
#
# Compute Neural Activity Index (NAI) of auditory primary cortex using MEG source activity
# of A1 and A2 (previously stored in a python data file)

import numpy as np
import matplotlib.pyplot as plt

# length of the array that contains simulated experiment (in synaptic timesteps)
synaptic_length=960

# total number of trials in simulated experiment
number_of_trials = 12

# name of the input file where MEG source activity is stored
MEG_source_file = 'meg_source_activity.npy'

# Load MEG source activity data files
MEG_source_activity = np.load(MEG_source_file)

# Extract number of timesteps from one of the matrices
timesteps = MEG_source_activity.shape

print timesteps

# Extract MEG source activity for A1 and A2 from synaptic array
a1_MEG_source = MEG_source_activity[0]
a2_MEG_source = MEG_source_activity[1]

# format the MEG source activity to eliminate timesteps not part of the experiment
a1_MEG_source = a1_MEG_source[1:synaptic_length+1]
a2_MEG_source = a2_MEG_source[1:synaptic_length+1]

# divide MEG source arrays into subarrays, each subarray containing exactly one experimental trial
a1_MEG_source_trials = np.split(a1_MEG_source, number_of_trials)
a2_MEG_source_trials = np.split(a2_MEG_source, number_of_trials)

# compute average of MEG activities across trials
a1_MEG_source_avg = np.mean(a1_MEG_source_trials, axis=0)
a2_MEG_source_avg = np.mean(a2_MEG_source_trials, axis=0)

# combine MEG source activity of A1 and A2 together
aud_MEG_source_activity = np.sum([a1_MEG_source_avg, a2_MEG_source_avg], axis=0)

print aud_MEG_source_activity.shape

# split average trial into S1 and S2 presentation, given that each trial is 80 synaptic timesteps long and it
# is divided up as follows:
#  1-20: Prior to S1 presentation
# 21-30: S1 presentation
# 31-50: Delay period
# 51:60: S2 presentation
# 61:80: Post stimulus presentation
# get rid of 10 timesteps at beginning of each trial and 10 timesteps
# at the end of each trial
aud_MEG_source_activity = aud_MEG_source_activity[10:70]

# split up the array in two equal parts to obtain auditory response to S1 and S2 separately.
# We therefore end up with NAI-S1 and NAI-S2 of 30 timesteps each.
# Which means that we each NAI is divided contains:
#      (a) 10 ts prior to S1/S2 onset,
#      (b) 10 ts for S1 / S2, and
#      (c) 10 ts after S1/S2 presentation
aud_MEG_source_activity = np.split(aud_MEG_source_activity, 2)

# Now, we are going to calculate an AER (Auditory Evoked Response) due to S1 and S2:
# (1) Normalize NAI activity by subtracting average baseline NAI activity:
normalized_NAI_S1 = aud_MEG_source_activity[0][10:20] - np.mean(aud_MEG_source_activity[0][0:9])
normalized_NAI_S2 = aud_MEG_source_activity[1][10:20] - np.mean(aud_MEG_source_activity[0][0:9])
# (1) integrate the NAI in a 350-ms window (stimulus presentation duration):
integrated_NAI_S1 = np.trapz(normalized_NAI_S1)
integrated_NAI_S2 = np.trapz(normalized_NAI_S2)
# (2) normalize the integrated NAI by subtracting average baseline NAI activity:
AER_S1 = integrated_NAI_S1
AER_S2 = integrated_NAI_S2

# Now, we are going to calculate modulation index (MI):
MI = ((AER_S1 - AER_S2) / (AER_S1 + AER_S2)) * 100.

print 'Modulation Index:', MI

# Set up figure to plot MEG source dynamics averaged across trials
fig = plt.figure(1)

plt.suptitle('MEG SOURCE DYNAMICS AVERAGED ACROSS TRIALS')

# Plot MEG signal
aud_plot_S1 = plt.plot(aud_MEG_source_activity[0], label='S1')
aud_plot_S2 = plt.plot(aud_MEG_source_activity[1], label='S2')

#plt.ylim(75, 120)

plt.legend()

# Show the plot on the screen
plt.show()

