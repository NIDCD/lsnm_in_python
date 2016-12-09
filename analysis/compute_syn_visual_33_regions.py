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
#   This file (compute_syn_visual_66_regions.py) was created on November 22, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on November 22, 2016
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_syn_visual_33_regions.py
#
# Calculate and plot simulated synaptic activities in 33 ROIs (33 in the right hemisphere)
# as defined by Haggman et al
# 
# ... using data from visual delay-match-to-sample simulation (or resting state simulation
# of the same duration as the DMS).
# It also saves the synaptic activities for 33 ROIs in a python data file
# (*.npy)
# The data is saved in a numpy array where the columns are the 33 ROIs
#

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# define the name of the output file where the integrated synaptic activity will be stored
syn_file = 'synaptic_in_33_ROIs.npy'

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the excel document:
#       "Location of visual LSNM modules within Connectome.xlsx"
# Extracted from The Virtual Brain Demo Data Sets

roi_dict = {
    'rLOF'  : range(  0,  19),    
    'rPORB' : range( 19,  25),          
    'rFP'   : range( 25,  27),          
    'rMOF'  : range( 27,  39),          
    'rPTRI' : range( 39,  47),          
    'rPOPE' : range( 47,  57),          
    'rRMF'  : range( 57,  79),          
    'rSF'   : range( 79, 125),          
    'rCMF'  : range(125, 138),          
    'rPREC' : range(138, 174),          
    'rPARC' : range(174, 186),          
    'rRAC'  : range(186, 190),          
    'rCAC'  : range(190, 194),          
    'rPC'   : range(194, 201),          
    'rISTC' : range(201, 209),          
    'rPSTC' : range(209, 240),          
    'rSMAR' : range(240, 256),          
    'rSP'   : range(256, 283),          
    'rIP'   : range(283, 311),          
    'rPCUN' : range(311, 334),          
    'rCUN'  : range(334, 344),          
    'rPCAL' : range(344, 354),          
    'rLOCC' : range(354, 373),          
    'rLING' : range(373, 390),          
    'rFUS'  : range(390, 412),          
    'rPARH' : range(412, 418),          
    'rENT'  : range(418, 420),          
    'rTP'   : range(420, 423),          
    'rIT'   : range(423, 442),          
    'rMT'   : range(442, 462),          
    'rBSTS' : range(462, 469),          
    'rST'   : range(469, 497),          
    'rTT'   : range(497, 500),
}

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_abs_syn.npy")

# create a numpy array of synaptic time-series, with a number of elements defined
# by the number of ROIs above and the number of time points in each synaptic time-series
synaptic = np.empty([len(roi_dict), tvb_synaptic.shape[0]])

# Load TVB host node synaptic activities into separate numpy arrays
idx=0
for roi in roi_dict:
    synaptic[idx] = np.sum(tvb_synaptic[:, 0, roi_dict[roi], 0] +   # excitatory unit of ROI
                           tvb_synaptic[:, 1, roi_dict[roi], 0],    # inhibitory unit of ROI
                           axis=1)
    idx = idx + 1
                         
# now, save all synaptic timeseries to a single file 
np.save(syn_file, synaptic)

# Extract total number of timesteps from synaptic time-series
timesteps = tvb_synaptic.shape[0]
print 'Timesteps = ', timesteps

# Construct a numpy array of timesteps (data points provided in data file)
# to convert from timesteps to time in seconds we do the following:
# Each simulation time-step equals 5 milliseconds
# However, we are recording only once every 10 time-steps
# Therefore, each data point in the output files represents 50 milliseconds.
# Thus, we need to multiply the datapoint times 50 ms...
# ... and divide by 1000 to convert to seconds
#t = np.linspace(0, 659*50./1000., num=660)
t = np.linspace(0, timesteps * 50.0 / 1000., num=timesteps)


# Set up figures to plot synaptic activity
plt.figure()
plt.suptitle('SIMULATED SYNAPTIC ACTIVITY OF ONE ROI')
plt.plot(t, synaptic[0])

# Show the plots on the screen
plt.show()
