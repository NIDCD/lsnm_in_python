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
#   This file (evaluate_performance.py) was created on September 27, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on September 27 2015  
# **************************************************************************/

# evaluate_performance.py
#
# Evaluates visual model response performance during a delayed match-to-sample
# task

import numpy as np
import matplotlib.pyplot as plt

# threshold for a single unit to be considered responding 'match'
# The number of single units firing is 'threshold_module'
# The threshold for a single unit to be considered firing is 'threshold'
# For example, below, we need 2 units firing at or above 0.6 to consider a match.
number_of_trials = 18
# threshold  = 0.7 THIS IS THE ONE THAT WORKS
# threshold module = 2 THIS IS THE ONE THAT WORKS
threshold = 0.7
threshold_module = 2

# Load response data file
exfr = np.loadtxt('exfr.out')

# Extract number of timesteps from one of the matrices
timesteps = exfr.shape[0]

# construct timeseries that include only response periods, separately for match
# and mismatch.
# The times here are based on the "script_to_replicate_Horwitz_2005_Fig_4_and_5.py"
# Change the times appropriately if using a different script.
# Only capture the state of the 81 units at the time interval between presentation
# of the second stimulus and just before the intertrial interval

correct_match = int(np.count_nonzero(np.amax(exfr[70:90] >= threshold, axis=0))  >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[290:310] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[730:750] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[950:970] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[1390:1410] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[1610:1630] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[2050:2070] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[2270:2290] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[2710:2730] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[2930:2950] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[3370:3390] >= threshold, axis=0)) >= threshold_module) + \
                int(np.count_nonzero(np.amax(exfr[3590:3610] >= threshold, axis=0)) >= threshold_module)

correct_nonmatch = int(np.count_nonzero(np.amax(exfr[180:200] >= threshold, axis=0)) < threshold_module) + \
                   int(np.count_nonzero(np.amax(exfr[840:860] >= threshold, axis=0)) < threshold_module) + \
                   int(np.count_nonzero(np.amax(exfr[1500:1520] >= threshold, axis=0)) < threshold_module) + \
                   int(np.count_nonzero(np.amax(exfr[2160:2180] >= threshold, axis=0)) < threshold_module) + \
                   int(np.count_nonzero(np.amax(exfr[2820:2840] >= threshold, axis=0)) < threshold_module) + \
                   int(np.count_nonzero(np.amax(exfr[3480:3500] >= threshold, axis=0)) < threshold_module)

correct_responses = correct_match + correct_nonmatch
pct_correct = correct_responses * 100. / number_of_trials

print '# Correct matches were:', correct_match
print '# Correct nonmatches were:', correct_nonmatch
print '# Correct responses were: ', correct_responses
print 'Performance was: ', pct_correct, '%'

# Contruct a numpy array of timesteps (data points provided in data file)
t = np.arange(0, timesteps, 1)

# Plot FR (Response module)
ax = plt.subplot()
ax.plot(t, exfr, color='r')
ax.set_yticks([0, 0.5, 1])
ax.set_xlim(0,timesteps)
plt.ylabel('R', rotation='horizontal', horizontalalignment='right')

plt.tight_layout()

# Show the plot on the screen
plt.show()

