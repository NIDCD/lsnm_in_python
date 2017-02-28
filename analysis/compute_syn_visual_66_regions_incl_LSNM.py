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
#   This file (compute_syn_visual_66_regions_incl_LSNM.py) was created on
#   February 14, 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on February 15, 2017
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_syn_visual_66_regions_incl_LSNM.py
#
# Calculate and plot simulated synaptic activities in 66 ROIs (33 in the right hemisphere
# and 33 in the left hemisphere), including LSNM modules within each of the 66 ROIs
#
# Note that the locations that were used to embed LSNM nodes is hardcoded in here, defined
# as a dictionary.
# 
# ... using data from visual delay-match-to-sample simulation (or resting state simulation
# of the same duration as the DMS).
# It also saves the synaptic activities for 66 ROIs in a python data file
# (*.npy)
# The data is saved in a numpy array where the columns are the 66 ROIs:
#

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

# set matplot lib parameters to produce visually appealing plots
mpl.style.use('ggplot')

# define the name of the output file where the integrated synaptic activity will be stored
syn_file = 'synaptic_in_66_ROIs_incl_LSNM.npy'

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
    'lLOF'  : range(500, 520),
    'lPORB' : range(520, 526),
    'lFP'   : range(526, 528),
    'lMOF'  : range(528, 540),
    'lPTRI' : range(540, 547),
    'lPOPE' : range(547, 558),
    'LRMF'  : range(558, 577),
    'lSF'   : range(577, 627),
    'lCMF'  : range(627, 640),
    'lPREC' : range(640, 676),
    'lPARC' : range(676, 687),
    'lRAC'  : range(687, 691),
    'lCAC'  : range(691, 695),
    'lPC'   : range(695, 702),
    'lISTC' : range(702, 710),
    'lPSTC' : range(710, 740),
    'lSMAR' : range(740, 759),
    'lSP'   : range(759, 786),
    'lIP'   : range(786, 811),
    'lPCUN' : range(811, 834),
    'lCUN'  : range(834, 842),
    'lPCAL' : range(842, 851),
    'lLOC'  : range(851, 873),
    'lLING' : range(873, 889),
    'lFUS'  : range(889, 911),
    'lPARH' : range(911, 917),
    'lENT'  : range(917, 920),
    'lTP'   : range(920, 924),
    'lIT'   : range(924, 941),
    'lMT'   : range(941, 960),
    'lBSTS' : range(960, 965),
    'lST'   : range(965, 994),
    'lTT'   : range(994, 998)
}

# the following dictionary defines the location of the LSNM nodes within the TVB connectome
# 66 (low resolution) ROIs

lsnm_tvb_link = {
    'rPCAL' : 'v1',
    'rFUS'  : 'v4',
    'rPARH' : 'it',
    'rPOPE' : 'fs',
    'rRMF'  : 'd1',
    'rPTRI' : 'd2',
    'rCMF'  : 'fr'
}

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_abs_syn.npy")

# Load LSNM synaptic activity data files into a numpy arrays
ev1h = np.loadtxt('ev1h_abs_syn.out')
ev1v = np.loadtxt('ev1v_abs_syn.out')
iv1h = np.loadtxt('iv1h_abs_syn.out')
iv1v = np.loadtxt('iv1v_abs_syn.out')
ev4h = np.loadtxt('ev4h_abs_syn.out')
ev4v = np.loadtxt('ev4v_abs_syn.out')
ev4c = np.loadtxt('ev4c_abs_syn.out')
iv4h = np.loadtxt('iv4h_abs_syn.out')
iv4v = np.loadtxt('iv4v_abs_syn.out')
iv4c = np.loadtxt('iv4c_abs_syn.out')
exss = np.loadtxt('exss_abs_syn.out')
inss = np.loadtxt('inss_abs_syn.out')
efd1 = np.loadtxt('efd1_abs_syn.out')
ifd1 = np.loadtxt('ifd1_abs_syn.out')
efd2 = np.loadtxt('efd2_abs_syn.out')
ifd2 = np.loadtxt('ifd2_abs_syn.out')
exfs = np.loadtxt('exfs_abs_syn.out')
infs = np.loadtxt('infs_abs_syn.out')
exfr = np.loadtxt('exfr_abs_syn.out')
infr = np.loadtxt('infr_abs_syn.out')

# add all LSNM units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
v1_syn_lsnm = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1)
v4_syn_lsnm = np.sum(ev4h + ev4v + ev4c + iv4h + iv4v + iv4c, axis = 1)
it_syn_lsnm = np.sum(exss + inss, axis = 1)
d1_syn_lsnm = np.sum(efd1 + ifd1, axis = 1)
d2_syn_lsnm = np.sum(efd2 + ifd2, axis = 1)
fs_syn_lsnm = np.sum(exfs + infs, axis = 1)
fr_syn_lsnm = np.sum(exfr + infr, axis = 1)

# create a numpy array of synaptic time-series, with a number of elements defined
# by the number of ROIs above and the number of time points in each synaptic time-series
synaptic = np.empty([len(roi_dict), tvb_synaptic.shape[0]])

# Load TVB host node synaptic activities into separate numpy arrays
idx=0
for roi in roi_dict:
    synaptic[idx] = np.sum(tvb_synaptic[:, 0, roi_dict[roi], 0] +   # excitatory unit of ROI
                           tvb_synaptic[:, 1, roi_dict[roi], 0],    # inhibitory unit of ROI
                           axis=1)
    # if the current ROI has an LSNM module embedded, then add it to synaptic sum
    if roi in lsnm_tvb_link:
        if   lsnm_tvb_link[roi] == 'v1':
            synaptic[idx] = synaptic[idx] + v1_syn_lsnm
        elif lsnm_tvb_link[roi] == 'v4':
            synaptic[idx] = synaptic[idx] + v4_syn_lsnm
        elif lsnm_tvb_link[roi] == 'it':
            synaptic[idx] = synaptic[idx] + it_syn_lsnm
        elif lsnm_tvb_link[roi] == 'd1':
            synaptic[idx] = synaptic[idx] + d1_syn_lsnm
        elif lsnm_tvb_link[roi] == 'd2':
            synaptic[idx] = synaptic[idx] + d2_syn_lsnm
        elif lsnm_tvb_link[roi] == 'fs':
            synaptic[idx] = synaptic[idx] + fs_syn_lsnm
        elif lsnm_tvb_link[roi] == 'fr':
            synaptic[idx] = synaptic[idx] + fr_syn_lsnm
    
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
