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
#   This file (compute_synaptic.py) was created on September 5, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on September 5 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_synaptic.py
#
# Calculate and plot simulated synaptic activities in given ROIs 
# 
# ... using data from visual delay-match-to-sample simulation.
# It also saves the synaptic activities for each and all modules in a python data file
# (*.npy)

import numpy as np

import matplotlib.pyplot as plt

# define the name of the output file where the BOLD timeseries will be stored
syn_file = 'synaptic_in_ROI_normalized.npy'

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the document:
#       "Hagmann's Brain Talairach Coordinates (obtained from Barry).doc"
# Provided by Barry Horwitz
# Please note that arrays in Python start from zero so one does to account for that and shift
# indices given by the above document by one location.
# Use all 10 nodes within rPCAL
v1_loc = range(344, 354)

# Use all 22 nodes within rFUS
v4_loc = range(390, 412)

# Use all 6 nodes within rPARH
it_loc = range(412, 418)

# Use all 22 nodes within rRMF
d1_loc =  range(57, 79)

# Use all nodes within rPTRI
d2_loc = range(39, 47)

# Use all nodes within rPOPE
fs_loc = range(47, 57)

# Use all nodes within rCMF
fr_loc = range(125, 138)

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_synaptic.npy")

# Load TVB host node synaptic activities into separate numpy arrays
tvb_ev1 = tvb_synaptic[:, 0, v1_loc[0]:v1_loc[-1]+1, 0]
tvb_ev4 = tvb_synaptic[:, 0, v4_loc[0]:v4_loc[-1]+1, 0]
tvb_eit = tvb_synaptic[:, 0, it_loc[0]:it_loc[-1]+1, 0]
tvb_ed1 = tvb_synaptic[:, 0, d1_loc[0]:d1_loc[-1]+1, 0]
tvb_ed2 = tvb_synaptic[:, 0, d2_loc[0]:d2_loc[-1]+1, 0]
tvb_efs = tvb_synaptic[:, 0, fs_loc[0]:fs_loc[-1]+1, 0]
tvb_efr = tvb_synaptic[:, 0, fr_loc[0]:fr_loc[-1]+1, 0]
tvb_iv1 = tvb_synaptic[:, 1, v1_loc[0]:v1_loc[-1]+1, 0]
tvb_iv4 = tvb_synaptic[:, 1, v4_loc[0]:v4_loc[-1]+1, 0]
tvb_iit = tvb_synaptic[:, 1, it_loc[0]:it_loc[-1]+1, 0]
tvb_id1 = tvb_synaptic[:, 1, d1_loc[0]:d1_loc[-1]+1, 0]
tvb_id2 = tvb_synaptic[:, 1, d2_loc[0]:d2_loc[-1]+1, 0]
tvb_ifs = tvb_synaptic[:, 1, fs_loc[0]:fs_loc[-1]+1, 0]
tvb_ifr = tvb_synaptic[:, 1, fr_loc[0]:fr_loc[-1]+1, 0]

# Load LSNM synaptic activity data files into a numpy arrays
ev1h = np.loadtxt('ev1h_synaptic.out')
ev1v = np.loadtxt('ev1v_synaptic.out')
iv1h = np.loadtxt('iv1h_synaptic.out')
iv1v = np.loadtxt('iv1v_synaptic.out')
ev4h = np.loadtxt('ev4h_synaptic.out')
ev4v = np.loadtxt('ev4v_synaptic.out')
ev4c = np.loadtxt('ev4c_synaptic.out')
iv4h = np.loadtxt('iv4h_synaptic.out')
iv4v = np.loadtxt('iv4v_synaptic.out')
iv4c = np.loadtxt('iv4c_synaptic.out')
exss = np.loadtxt('exss_synaptic.out')
inss = np.loadtxt('inss_synaptic.out')
efd1 = np.loadtxt('efd1_synaptic.out')
ifd1 = np.loadtxt('ifd1_synaptic.out')
efd2 = np.loadtxt('efd2_synaptic.out')
ifd2 = np.loadtxt('ifd2_synaptic.out')
exfs = np.loadtxt('exfs_synaptic.out')
infs = np.loadtxt('infs_synaptic.out')
exfr = np.loadtxt('exfr_synaptic.out')
infr = np.loadtxt('infr_synaptic.out')

# add all units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
v1_syn = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1) + np.sum(tvb_ev1+tvb_iv1, axis=1)
v4_syn = np.sum(ev4h + ev4v + ev4c + iv4h + iv4v + iv4c, axis = 1) + np.sum(tvb_ev4+tvb_iv4, axis=1)
it_syn = np.sum(exss + inss, axis = 1) + np.sum(tvb_eit+tvb_iit, axis=1)
d1_syn = np.sum(efd1 + ifd1, axis = 1) + np.sum(tvb_ed1+tvb_id1, axis=1)
d2_syn = np.sum(efd2 + ifd2, axis = 1) + np.sum(tvb_ed2+tvb_id2, axis=1)
fs_syn = np.sum(exfs + infs, axis = 1) + np.sum(tvb_efs+tvb_ifs, axis=1)
fr_syn = np.sum(exfr + infr, axis = 1) + np.sum(tvb_efr+tvb_ifr, axis=1)

# get rid of the first time point ('zero point') bc it could skew correlations later
v1_syn = np.delete(v1_syn, 0)
v4_syn = np.delete(v4_syn, 0)
it_syn = np.delete(it_syn, 0)
d1_syn = np.delete(d1_syn, 0)
d2_syn = np.delete(d2_syn, 0)
fs_syn = np.delete(fs_syn, 0)
fr_syn = np.delete(fr_syn, 0)

# ...and normalize the synaptic activities of each module (convert to percentage signal change)
v1_syn = v1_syn / np.mean(v1_syn) * 100. - 100.
v4_syn = v4_syn / np.mean(v4_syn) * 100. - 100.
it_syn = it_syn / np.mean(it_syn) * 100. - 100.
d1_syn = d1_syn / np.mean(d1_syn) * 100. - 100.
d2_syn = d2_syn / np.mean(d2_syn) * 100. - 100.
fs_syn = fs_syn / np.mean(fs_syn) * 100. - 100.
fr_syn = fr_syn / np.mean(fr_syn) * 100. - 100.

# create a numpy array of timeseries
synaptic = np.array([v1_syn, v4_syn, it_syn, fs_syn, d1_syn, d2_syn, fr_syn])

# now, save all BOLD timeseries to a single file 
np.save(syn_file, synaptic)

# Set up figure to plot synaptic activity
plt.figure()

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot synaptic activities
plt.plot(v1_syn)
plt.plot(it_syn)
plt.plot(d1_syn)

# Show the plots on the screen
plt.show()
