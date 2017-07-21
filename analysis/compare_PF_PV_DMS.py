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
#   This file (compare_PF_PV_DMS.py) was created on July 10 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on July 20 2017
#
#   Based on computer code originally developed by Barry Horwitz et al
#   Also based on Python2.7 tutorials
#
# Python function to convert adjacency matrix to edge list
# adapted from Python code by: Jermaine Kaminski
# From Github repository (under MIT License)
# https://github.com/jermainkaminski/Adjacency-Matrix-to-Edge-List

# **************************************************************************/
#
# compare_PF_PV_DMS.py
#
# Reads the correlation coefficients (functional connectivity matrix) from
# several python (*.npy) data files, each
# corresponding to a single subject, and calculates:
# (1) the average functional connectivity across all subjects for four conditions:
#     (a) TVB-only Resting State,
#     (b) TVB/LSNM Resting State,
#     (c) TVB/LSNM Passive Viewing, and
#     (d) TVB/LSNM Delayed Match-to-sample task,
# (2) means, standard deviations and variances for each data point.
# Means, standard deviations, and variances are stored in a text output file.
# For the calculations, It uses
# previously calculated cross-correlation coefficients using simulated BOLD
# in Hagmann's connectome 66 ROIs (right and left hemispheres)
# It also performs a paired t-test for the comparison between the mean of each condition
# (task-based vs resting-state), and displays the t values. Note that in a paired t-test,
# each subject in the study is in both the treatment and the control group.

################## TESTING 3D PLOT OF ADJACENCY MATRIX
#from tvb.simulator.lab import connectivity
#from mayavi import mlab
################## TESTING 3D PLOT OF ADJACENCY MATRIX

from tvb.datatypes import connectivity

import numpy as np
import matplotlib.pyplot as plt

#import plotly.plotly as py
#import plotly.tools as tls

import matplotlib as mpl

import pandas as pd

from scipy.stats import t

from scipy.stats import itemfreq

from scipy import stats

import math as m

from scipy.stats import kurtosis

from scipy.stats import skew

import scipy.io

from matplotlib import cm as CM

from mne.viz import circular_layout, plot_connectivity_circle

import bct as bct

import csv

from nipy.modalities.fmri.glm import GeneralLinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx

# The following converts from adjacency matrix to edge list
# Adapted from code by: Jermaine Kaminski
# From Github repository (under MIT License)
# https://github.com/jermainkaminski/Adjacency-Matrix-to-Edge-List
def adj_to_list(input_filename,output_filename,delimiter):
    '''Takes the adjacency matrix on file input_filename into a list of edges and saves it into output_filename'''
    A=pd.read_csv(input_filename,delimiter=delimiter,index_col=0)
    List=[]
    for source in A.index.values:
        for target in A.index.values:
            if A[source][target]==1:
                List.append((target,source))
    with open(output_filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(List)
    return List
# end of adj_to_list

# set matplot lib parameters to produce visually appealing plots
#mpl.style.use('ggplot')

# construct array of indices of modules contained in Hagmann's connectome
# (right hemisphere)
ROIs = np.arange(66)

# construct array of subjects to be considered
subjects = np.arange(10)

# define name of input file where Hagmann empirical data is stored (matlab file given to us
# by Olaf Sporns and Chris Honey
hagmann_data = '../../../HAGMANNS_DSI_DATA/DSI_release2_2011.mat'

# define output file where Functional connectivity averages will be stored
EMP_RS_FC_file = 'emp_rs_fc.npy'
TVB_RS_FC_avg_file = 'tvb_rs_fc_avg.npy'
TVB_LSNM_RS_FC_avg_file = 'tvb_lsnm_rs_fc_avg.npy'
TVB_LSNM_PV_FC_avg_file = 'tvb_lsnm_pv_fc_avg.npy'
TVB_LSNM_DMS_FC_avg_file= 'tvb_lsnm_dms_fc_avg.npy'
PV_minus_RS_file = 'pv_minus_rs.npy'
DMS_minus_RS_file = 'dms_minus_rs.npy'
TB_FC_avg_file = 'tb_fc_avg.npy'

# declare ROI labels
labels =  [' rLOF',
           'rPORB',         
           '  rFP',          
           ' rMOF',          
           'rPTRI',          
           'rPOPE',          
           ' rRMF',          
           '  rSF',          
           ' rCMF',          
           'rPREC',          
           'rPARC',          
           ' rRAC',          
           ' rCAC',          
           '  rPC',          
           'rISTC',          
           'rPSTC',          
           'rSMAR',          
           '  rSP',          
           '  rIP',          
           'rPCUN',          
           ' rCUN',          
           'rPCAL',          
           'rLOCC',          
           'rLING',          
           ' rFUS',          
           'rPARH',          
           ' rENT',          
           '  rTP',          
           '  rIT',          
           '  rMT',          
           'rBSTS',          
           '  rST',          
           '  rTT',
           ' lLOF',
           'lPORB',
           '  lFP',
           ' lMOF',
           'lPTRI',
           'lPOPE',
           ' lRMF',
           '  lSF',
           ' lCMF',
           'lPREC',
           'lPARC',
           ' lRAC',
           ' lCAC',
           '  lPC',
           'lISTC',
           'lPSTC',
           'lSMAR',
           '  lSP',
           '  lIP',
           'lPCUN',
           ' lCUN',
           'lPCAL',
           'lLOCC',
           'lLING',
           ' lFUS',
           'lPARH',
           ' lENT',
           '  lTP',
           '  lIT',
           '  lMT',
           'lBSTS',
           '  lST',
           '  lTT'
]

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

# declares the indexes of those brain regions in the low resolutio ROI array that have direct connections
# with the embedded LSNM nodes
connected=[0,1,4,5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,23,24,25,28,29,30,31,40,43,45,46,53,54,56]


# declares membership of each one of the 66 areas above in one of six modules as defined by Hagmann et al (2008)
# See Table S2 from Supplementary material from Lee et al, 2016, Neuroimage
CI = np.array([6,6,6,6,6,6,6,6,6,6,2,6,2,2,2,4,4,4,4,2,1,1,4,1,4,4,4,4,4,4,4,4,4,
               5,5,5,5,5,5,5,5,5,5,2,5,2,2,2,3,3,3,3,1,1,1,3,1,3,1,3,3,3,3,3,3,3])

# define the names of the input files where the correlation coefficients were stored
TVB_RS_subj1  = 'subject_tvb/output.RestingState_01/xcorr_matrix_998_regions.npy'
TVB_RS_subj2  = 'subject_tvb/output.RestingState_02/xcorr_matrix_998_regions.npy'
TVB_RS_subj3  = 'subject_tvb/output.RestingState_03/xcorr_matrix_998_regions.npy'
TVB_RS_subj4  = 'subject_tvb/output.RestingState_04/xcorr_matrix_998_regions.npy'
TVB_RS_subj5  = 'subject_tvb/output.RestingState_05/xcorr_matrix_998_regions.npy'
TVB_RS_subj6  = 'subject_tvb/output.RestingState_06/xcorr_matrix_998_regions.npy'
TVB_RS_subj7  = 'subject_tvb/output.RestingState_07/xcorr_matrix_998_regions.npy'
TVB_RS_subj8  = 'subject_tvb/output.RestingState_08/xcorr_matrix_998_regions.npy'
TVB_RS_subj9  = 'subject_tvb/output.RestingState_09/xcorr_matrix_998_regions.npy'
TVB_RS_subj10 = 'subject_tvb/output.RestingState_10/xcorr_matrix_998_regions.npy'

TVB_LSNM_RS_subj1  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj2  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj3  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj4  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj5  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj6  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj7  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj8  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj9  = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_RS_subj10 = 'subject_12/output.Fixation_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'

TVB_LSNM_PV_subj1  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj2  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj3  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj4  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj5  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj6  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj7  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj8  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj9  = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_PV_subj10 = 'subject_12/output.PassiveViewing_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'

TVB_LSNM_DMS_subj1  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj2  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj3  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj4  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj5  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj6  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj7  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj8  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj9  = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'
TVB_LSNM_DMS_subj10 = 'subject_12/output.DMSTask_incl_PreSMA/xcorr_matrix_998_regions_syn.npy'

# upload fMRI BOLD time series for all subject and conditions
TVB_LSNM_PF_BOLD = 'subject_12/output.Fixation_incl_PreSMA/bold_balloon_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PV_BOLD = 'subject_12/output.PassiveViewing_incl_PreSMA/bold_balloon_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_BOLD= 'subject_12/output.DMSTask_incl_PreSMA/bold_balloon_998_regions_3T_0.25Hz.npy'

# number of randomly generated functional connectivity matrices to be generated. These FC matrices
# will be used to compare against actual FC matrices. Used 20 found in the graph theory literature.
num_random = 20

hires_ROIs = 998 
lores_ROIs = 66

# Define structural connectivity that was used for simulations (998 ROI matrix from TVB demo set)
print 'Loading TVB structural connectivity used for simulations'
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
empirical_sc_hires = np.asarray(white_matter.weights)[0:hires_ROIs, 0:hires_ROIs]

# switch to a prettier color palette
#plt.style.use('ggplot')

##################################################################################
# Open input files containing both empirical and simulated functional connectivity
# matrices
##################################################################################
print 'Opening input files containing FC matrices (empirical and simulated)...'

# open matlab file that contains hagmann empirical data
hagmann_empirical = scipy.io.loadmat(hagmann_data)

# open files that contain correlation coefficients
tvb_rs_subj1  = np.load(TVB_RS_subj1)
tvb_rs_subj2  = np.load(TVB_RS_subj2)
tvb_rs_subj3  = np.load(TVB_RS_subj3)
tvb_rs_subj4  = np.load(TVB_RS_subj4)
tvb_rs_subj5  = np.load(TVB_RS_subj5)
tvb_rs_subj6  = np.load(TVB_RS_subj6)
tvb_rs_subj7  = np.load(TVB_RS_subj7)
tvb_rs_subj8  = np.load(TVB_RS_subj8)
tvb_rs_subj9  = np.load(TVB_RS_subj9)
tvb_rs_subj10 = np.load(TVB_RS_subj10)

tvb_lsnm_rs_subj1  = np.load(TVB_LSNM_RS_subj1)
tvb_lsnm_rs_subj2  = np.load(TVB_LSNM_RS_subj2)
tvb_lsnm_rs_subj3  = np.load(TVB_LSNM_RS_subj3)
tvb_lsnm_rs_subj4  = np.load(TVB_LSNM_RS_subj4)
tvb_lsnm_rs_subj5  = np.load(TVB_LSNM_RS_subj5)
tvb_lsnm_rs_subj6  = np.load(TVB_LSNM_RS_subj6)
tvb_lsnm_rs_subj7  = np.load(TVB_LSNM_RS_subj7)
tvb_lsnm_rs_subj8  = np.load(TVB_LSNM_RS_subj8)
tvb_lsnm_rs_subj9  = np.load(TVB_LSNM_RS_subj9)
tvb_lsnm_rs_subj10 = np.load(TVB_LSNM_RS_subj10)

tvb_lsnm_pv_subj1  = np.load(TVB_LSNM_PV_subj1)
tvb_lsnm_pv_subj2  = np.load(TVB_LSNM_PV_subj2)
tvb_lsnm_pv_subj3  = np.load(TVB_LSNM_PV_subj3)
tvb_lsnm_pv_subj4  = np.load(TVB_LSNM_PV_subj4)
tvb_lsnm_pv_subj5  = np.load(TVB_LSNM_PV_subj5)
tvb_lsnm_pv_subj6  = np.load(TVB_LSNM_PV_subj6)
tvb_lsnm_pv_subj7  = np.load(TVB_LSNM_PV_subj7)
tvb_lsnm_pv_subj8  = np.load(TVB_LSNM_PV_subj8)
tvb_lsnm_pv_subj9  = np.load(TVB_LSNM_PV_subj9)
tvb_lsnm_pv_subj10 = np.load(TVB_LSNM_PV_subj10)

tvb_lsnm_dms_subj1  = np.load(TVB_LSNM_DMS_subj1)
tvb_lsnm_dms_subj2  = np.load(TVB_LSNM_DMS_subj2)
tvb_lsnm_dms_subj3  = np.load(TVB_LSNM_DMS_subj3)
tvb_lsnm_dms_subj4  = np.load(TVB_LSNM_DMS_subj4)
tvb_lsnm_dms_subj5  = np.load(TVB_LSNM_DMS_subj5)
tvb_lsnm_dms_subj6  = np.load(TVB_LSNM_DMS_subj6)
tvb_lsnm_dms_subj7  = np.load(TVB_LSNM_DMS_subj7)
tvb_lsnm_dms_subj8  = np.load(TVB_LSNM_DMS_subj8)
tvb_lsnm_dms_subj9  = np.load(TVB_LSNM_DMS_subj9)
tvb_lsnm_dms_subj10 = np.load(TVB_LSNM_DMS_subj10)

#####################################################################
# open files that contain 998-node BOLD timeseries for all conditions
#####################################################################
tvb_lsnm_pf_bold  = np.load(TVB_LSNM_PF_BOLD)
tvb_lsnm_pv_bold  = np.load(TVB_LSNM_PV_BOLD)
tvb_lsnm_dms_bold = np.load(TVB_LSNM_DMS_BOLD)

# construct numpy arrays that contain correlation coefficient arrays for all subjects
tvb_rs = np.array([tvb_rs_subj1, tvb_rs_subj2, tvb_rs_subj3,
                       tvb_rs_subj4, tvb_rs_subj5, tvb_rs_subj6,
                       tvb_rs_subj7, tvb_rs_subj8, tvb_rs_subj9,
                       tvb_rs_subj10 ]) 
tvb_lsnm_rs = np.array([tvb_lsnm_rs_subj1, tvb_lsnm_rs_subj2, tvb_lsnm_rs_subj3,
                       tvb_lsnm_rs_subj4, tvb_lsnm_rs_subj5, tvb_lsnm_rs_subj6,
                       tvb_lsnm_rs_subj7, tvb_lsnm_rs_subj8, tvb_lsnm_rs_subj9,
                       tvb_lsnm_rs_subj10 ]) 
tvb_lsnm_pv = np.array([tvb_lsnm_pv_subj1, tvb_lsnm_pv_subj2, tvb_lsnm_pv_subj3,
                       tvb_lsnm_pv_subj4, tvb_lsnm_pv_subj5, tvb_lsnm_pv_subj6,
                       tvb_lsnm_pv_subj7, tvb_lsnm_pv_subj8, tvb_lsnm_pv_subj9,
                       tvb_lsnm_pv_subj10 ]) 
tvb_lsnm_dms = np.array([tvb_lsnm_dms_subj1, tvb_lsnm_dms_subj2, tvb_lsnm_dms_subj3,
                       tvb_lsnm_dms_subj4, tvb_lsnm_dms_subj5, tvb_lsnm_dms_subj6,
                       tvb_lsnm_dms_subj7, tvb_lsnm_dms_subj8, tvb_lsnm_dms_subj9,
                       tvb_lsnm_dms_subj10 ]) 

# calculate number of subjects from one of the arrays above
num_of_sub = len(tvb_rs)

# num_of_density thresholds to be used
num_of_densities = 35
#num_of_densities = 100

############################################################################
# compress BOLD timeseries of 998 nodes for each condition into corresponding
# 66 nodes arrays
############################################################################
print 'Compressing hi-res 998-node BOLD timeseries into 66-node BOLD timeseries...'

# extract length of BOLD timeseries from one of the BOLD arrays
ts_length = tvb_lsnm_pf_bold.shape[1]
t_steps = np.arange(ts_length)

# create a numpy array of BOLD time-series, with a number of elements defined
# by the number of ROIs in the low resolution scheme (i.e., 66) and the number
# of time points in each synaptic time-series
tvb_lsnm_pf_bold_lowres  = np.zeros([len(roi_dict), tvb_lsnm_pf_bold.shape[1]])
tvb_lsnm_pv_bold_lowres  = np.zeros([len(roi_dict), tvb_lsnm_pv_bold.shape[1]])
tvb_lsnm_dms_bold_lowres = np.zeros([len(roi_dict), tvb_lsnm_dms_bold.shape[1]])

idx=0
for roi in roi_dict:
    for t in t_steps:
        tvb_lsnm_pf_bold_lowres[idx, t]  = np.mean(tvb_lsnm_pf_bold[roi_dict[roi], t])
        tvb_lsnm_pv_bold_lowres[idx, t]  = np.mean(tvb_lsnm_pv_bold[roi_dict[roi], t])
        tvb_lsnm_dms_bold_lowres[idx, t] = np.mean(tvb_lsnm_dms_bold[roi_dict[roi], t])
    idx = idx + 1                         

print 'Low resolution BOLD array have the following dimensions: ', tvb_lsnm_pf_bold_lowres.shape

############################################################################
# compress 998x998 mean simulated FC matrix to 66x66 FC matrix by averaging
# ... and do this for each one of the given subjects
############################################################################
print 'Compressing hi-res FC matrices into low-res FC matrices...'

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
tvb_rs_z        = np.arctanh(tvb_rs[:,0:hires_ROIs, 0:hires_ROIs])
tvb_lsnm_rs_z   = np.arctanh(tvb_lsnm_rs[:,0:hires_ROIs, 0:hires_ROIs])
tvb_lsnm_pv_z   = np.arctanh(tvb_lsnm_pv[:,0:hires_ROIs, 0:hires_ROIs])
tvb_lsnm_dms_z  = np.arctanh(tvb_lsnm_dms[:,0:hires_ROIs, 0:hires_ROIs])

# initialize 66x66 matrices
tvb_rs_lowres_Z = np.zeros([num_of_sub, lores_ROIs, lores_ROIs])
tvb_lsnm_rs_lowres_Z = np.zeros([num_of_sub, lores_ROIs, lores_ROIs])
tvb_lsnm_pv_lowres_Z = np.zeros([num_of_sub, lores_ROIs, lores_ROIs])
tvb_lsnm_dms_lowres_Z= np.zeros([num_of_sub, lores_ROIs, lores_ROIs])

# subtract 1 from empirical labels array bc numpy arrays start with zero
hagmann_empirical['roi_lbls'][0] = hagmann_empirical['roi_lbls'][0] - 1

# count the number of times each lowres label appears in the hires matrix
freq_array = itemfreq(hagmann_empirical['roi_lbls'][0])

for s in range(0, num_of_sub):
    for i in range(0, hires_ROIs):
        for j in range(0, hires_ROIs):

            # extract low-res coordinates from hi-res empirical labels matrix
            x = hagmann_empirical['roi_lbls'][0][i]
            y = hagmann_empirical['roi_lbls'][0][j]
            tvb_rs_lowres_Z[s, x, y]       += tvb_rs_z[s, i, j]
            tvb_lsnm_rs_lowres_Z[s, x, y]  += tvb_lsnm_rs_z[s, i, j]
            tvb_lsnm_pv_lowres_Z[s, x, y]  += tvb_lsnm_pv_z[s, i, j]
            tvb_lsnm_dms_lowres_Z[s, x, y] += tvb_lsnm_dms_z[s, i, j]

    # average the sum of each bucket dividing by no. of items in each bucket
    for x in range(0, lores_ROIs):
        for y in range(0, lores_ROIs):
            total_freq = freq_array[x][1] * freq_array[y][1]
            tvb_rs_lowres_Z[s, x, y] =      tvb_rs_lowres_Z[s, x, y] / total_freq
            tvb_lsnm_rs_lowres_Z[s, x, y] = tvb_lsnm_rs_lowres_Z[s, x, y] / total_freq 
            tvb_lsnm_pv_lowres_Z[s, x, y] = tvb_lsnm_pv_lowres_Z[s, x, y] / total_freq
            tvb_lsnm_dms_lowres_Z[s, x, y]= tvb_lsnm_dms_lowres_Z[s, x, y] / total_freq

###########################################################################
# Compress hi-res empirical SC & FC matrices into lo-res SC & FC matrices
# we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
###########################################################################
empirical_fc_hires_Z = np.arctanh(hagmann_empirical['COR_fMRI_average'][0:hires_ROIs, 0:hires_ROIs])

# initialize 66x66 matrix
empirical_fc_lowres_Z = np.zeros([lores_ROIs, lores_ROIs])
empirical_sc_lowres   = np.zeros([lores_ROIs, lores_ROIs])

# compress 998x998 FC matrix to 66x66 FC matrix by averaging
for i in range(0, hires_ROIs):
    for j in range(0, hires_ROIs):

        # extract low-res coordinates from hi-res labels matrix
        x = hagmann_empirical['roi_lbls'][0][i]
        y = hagmann_empirical['roi_lbls'][0][j]

        empirical_fc_lowres_Z[x, y] += empirical_fc_hires_Z[i, j]
        empirical_sc_lowres[x, y]   += empirical_sc_hires[i, j]

# divide each sum by the number of hires ROIs within each lowres ROI
for x in range(0, lores_ROIs):
    for y in range(0, lores_ROIs):
        total_freq = freq_array[x][1] * freq_array[y][1]
        empirical_fc_lowres_Z[x, y] = empirical_fc_lowres_Z[x, y] / total_freq
        empirical_sc_lowres[x, y]   = empirical_sc_lowres[x, y] / total_freq

# now, convert back to from Z to R correlation coefficients
empirical_fc_hires  = np.tanh(empirical_fc_hires_Z)
print 'Shape of Empirical FC Matrix: ', empirical_fc_hires.shape
empirical_fc_lowres = np.tanh(empirical_fc_lowres_Z)
print 'Shape of Empirical FC Matrix (low-res): ', empirical_fc_lowres.shape

#############################################################################
# Average simuated FC matrices across subjects
#############################################################################
print 'Averaging FC matrices across subjects...'

# calculate the mean of correlation coefficients across all given subjects
# for the high resolution (998x998) correlation matrices
tvb_rs_z_mean = np.mean(tvb_rs_z, axis=0)
tvb_lsnm_rs_z_mean = np.mean(tvb_lsnm_rs_z, axis=0)
tvb_lsnm_pv_z_mean = np.mean(tvb_lsnm_pv_z, axis=0)
tvb_lsnm_dms_z_mean = np.mean(tvb_lsnm_dms_z, axis=0)

# calculate the mean of correlation coefficients across all given subjects
# for the low resolution (66x66) correlation matrices
tvb_rs_lowres_Z_mean = np.mean(tvb_rs_lowres_Z, axis=0)
tvb_lsnm_rs_lowres_Z_mean = np.mean(tvb_lsnm_rs_lowres_Z, axis=0)
tvb_lsnm_pv_lowres_Z_mean = np.mean(tvb_lsnm_pv_lowres_Z, axis=0)
tvb_lsnm_dms_lowres_Z_mean = np.mean(tvb_lsnm_dms_lowres_Z, axis=0)

############################################################################
# Calculate differences between RS and PV and RS and DMS mean FC matrices
############################################################################
print 'Calculating differences between RS and Task matrices...'

pv_minus_rs_Z =  tvb_lsnm_pv_lowres_Z_mean  - tvb_lsnm_rs_lowres_Z_mean
dms_minus_rs_Z = tvb_lsnm_dms_lowres_Z_mean - tvb_lsnm_rs_lowres_Z_mean
dms_minus_pv_Z = tvb_lsnm_dms_lowres_Z_mean - tvb_lsnm_pv_lowres_Z_mean

# calculate the standard error of the mean of correlation coefficients across subjects
#fc_rs_z_std = np.std(fc_rs_z, axis=0)
#fc_tb_z_std = np.std(fc_tb_z, axis=0)

# calculate the variance of the correlation coefficients across subjects
#fc_rs_z_var = np.var(fc_rs_z, axis=0)
#fc_tb_z_var = np.var(fc_tb_z, axis=0)

# Calculate the statistical significance by using a two-tailed paired t-test:
# We are going to have one group of 10 subjects, doing DMS task(TB) and Resting State (RS)
# STEPS:
#     (1) Set up hypotheses:
# The NULL hypothesis is:
#          * The mean difference between paired observations (TB and RS) is zero
#            In other words, H_0 : mean(TB) = mean(RS)
# Our alternative hypothesis is:
#          * The mean difference between paired observations (TB and RS) is not zero
#            In other words, H_A : mean(TB) =! mean(RS)
#     (2) Set a significance level:
#         alpha = 1 - confidence interval = 1 - 95% = 1 - 0.95 = 0.05 
#alpha = 0.05
#     (3) What is the critical value and the rejection region?
#n = 10                         # sample size
#df = n  - 1                    # degrees-of-freedom = n minus 1
                               # sample size is 10 because there are 10 subjects in each condition
#rejection_region = 2.262       # as found on t-test table for t and dof given,
                               # null-hypothesis (H_0) will be rejected for those t above rejection_region
#     (4) compute the value of the test statistic                               
# calculate differences between the pairs of data:
#d  = fc_tb_z - fc_rs_z
# calculate the mean of those differences
#d_mean = np.mean(d, axis=0)
# calculate the standard deviation of those differences
#d_std = np.std(d, axis=0)
# calculate square root of sample size
#sqrt_n = m.sqrt(n)
# calculate standard error of the mean differences
#d_sem = d_std/sqrt_n 
# calculate the t statistic:
#t_star = d_mean / d_sem

# threshold and binarize the array of t statistics:
#t_star_mask = np.where(t_star>rejection_region, 0, 1)

#initialize new figure for t-test values
#fig = plt.figure('T-test values')
#ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
#mask = np.tri(t_star.shape[0], k=0)
#mask = np.transpose(mask)
#t_star = np.ma.array(t_star, mask=mask)          # mask out upper triangle
#t_star = np.ma.array(t_star, mask=t_star_mask)   # mask out elements rejected by t-test

# plot correlation matrix as a heatmap
#cmap = CM.get_cmap('jet', 10)
#cmap.set_bad('w')
#cax = ax.imshow(t_star, interpolation='nearest', cmap=cmap)
#ax.grid(False)
#plt.colorbar(cax)

# now, convert back to from Z to R correlation coefficients (low-res)
tvb_rs_lowres_mean  = np.tanh(tvb_rs_lowres_Z_mean)
tvb_lsnm_lowres_rs_mean  = np.tanh(tvb_lsnm_rs_lowres_Z_mean)
tvb_lsnm_lowres_pv_mean  = np.tanh(tvb_lsnm_pv_lowres_Z_mean)
tvb_lsnm_lowres_dms_mean  = np.tanh(tvb_lsnm_dms_lowres_Z_mean)

# now, convert back to from Z to R correlation coefficients (hi-res)
tvb_rs_mean  = np.tanh(tvb_rs_z_mean)
tvb_lsnm_rs_mean  = np.tanh(tvb_lsnm_rs_z_mean)
tvb_lsnm_pv_mean  = np.tanh(tvb_lsnm_pv_z_mean)
tvb_lsnm_dms_mean  = np.tanh(tvb_lsnm_dms_z_mean)

# also convert matrices of differences to R correlation coefficients 
pv_minus_rs  = np.tanh(pv_minus_rs_Z)
dms_minus_rs = np.tanh(dms_minus_rs_Z)
dms_minus_pv = np.tanh(dms_minus_pv_Z)

# fill diagonals of all matrices with zeros
np.fill_diagonal(empirical_fc_lowres, 0)
np.fill_diagonal(tvb_rs_lowres_mean, 0)
np.fill_diagonal(tvb_lsnm_lowres_rs_mean, 0)
np.fill_diagonal(tvb_lsnm_lowres_pv_mean, 0)
np.fill_diagonal(tvb_lsnm_lowres_dms_mean, 0)
np.fill_diagonal(tvb_rs_mean, 0)
np.fill_diagonal(tvb_lsnm_rs_mean, 0)
np.fill_diagonal(tvb_lsnm_pv_mean, 0)
np.fill_diagonal(tvb_lsnm_dms_mean, 0)
np.fill_diagonal(pv_minus_rs, 0)
np.fill_diagonal(dms_minus_rs, 0)
np.fill_diagonal(dms_minus_pv, 0)

# calculate the max and min of FC matrices and store in variables for later use
FC_max = max([np.amax(tvb_rs_lowres_mean),
              np.amax(tvb_lsnm_lowres_rs_mean),
              np.amax(tvb_lsnm_lowres_pv_mean),
              np.amax(tvb_lsnm_lowres_dms_mean)])
FC_diff_max = max([np.amax(pv_minus_rs),
                   np.amax(dms_minus_rs),
                   np.amax(dms_minus_pv)])
FC_emp_max = np.amax(empirical_fc_lowres)
FC_min = min([np.amin(tvb_rs_lowres_mean),
              np.amin(tvb_lsnm_lowres_rs_mean),
              np.amin(tvb_lsnm_lowres_pv_mean),
              np.amin(tvb_lsnm_lowres_dms_mean)])
FC_diff_min = max([np.amin(pv_minus_rs),
                   np.amin(dms_minus_rs),
                   np.amin(dms_minus_pv)])
FC_emp_min = np.amin(empirical_fc_lowres)
FC_max_diff_pv_rs  = np.sum(np.absolute(pv_minus_rs), axis=1)
FC_max_diff_dms_rs = np.sum(np.absolute(dms_minus_rs), axis=1)
FC_max_diff_idx_pv_rs  = np.where(FC_max_diff_pv_rs  == np.amax(FC_max_diff_pv_rs))
FC_max_diff_idx_dms_rs = np.where(FC_max_diff_dms_rs == np.amax(FC_max_diff_dms_rs))
print 'Minimum correlation coefficient in FC matrices was: ', FC_min
print 'Maximum correlation coefficient in FC matrices was: ', FC_max
print 'Minumum correlation coefficient in FC diff matrices: ', FC_diff_min
print 'Maximum correlation coefficient in FC diff matrices" ', FC_diff_max
print 'Minimum correlation coefficient in Empirical FC matrices: ', FC_emp_min
print 'Maximum correlation coefficient in Empirical FC matrices: ', FC_emp_max
print 'Maximum cummulative (DMS - RS) abs. difference is at row ', FC_max_diff_idx_dms_rs[0], ', ROI: ', labels[FC_max_diff_idx_dms_rs[0]] 
print 'Maximum cummulative ( PV - RS) abs. difference is at row ', FC_max_diff_idx_pv_rs[0], ',  ROI: ', labels[FC_max_diff_idx_pv_rs[0]] 

# save FC matrices to outputfiles
print 'Saving FC matrices for later use...'
np.save(EMP_RS_FC_file, empirical_fc_lowres)
np.save(TVB_RS_FC_avg_file, tvb_rs_lowres_mean)
np.save(TVB_LSNM_RS_FC_avg_file, tvb_lsnm_lowres_rs_mean)
np.save(TVB_LSNM_PV_FC_avg_file, tvb_lsnm_lowres_pv_mean)
np.save(TVB_LSNM_DMS_FC_avg_file, tvb_lsnm_lowres_dms_mean)
np.save(PV_minus_RS_file,  pv_minus_rs )
np.save(DMS_minus_RS_file, dms_minus_rs)

##################################################################################
# Compute graph theoretical metrics on randomly generated matices and on
# FC matrices across conditions, using a range of
# different sparsity (threshold) levels
# Graph theory metrics computed are: Global efficiency, clustering, and modularity,
# and nodal degree.
#################################################################################

# threshold the connectivity matrix to preserve only a proportion 'p' of
# the strongest weights, then binarize the matrix
min_sparsity = 0.06
max_sparsity = 0.4
#min_sparsity = 0.01
#max_sparsity = 1.0

num_sparsity = num_of_densities
EMP_RS_EFFICIENCY_G   = np.zeros(num_sparsity)
EMP_RS_EFFICIENCY_L = np.zeros(num_sparsity)
EMP_RS_CLUSTERING = np.zeros(num_sparsity)
EMP_RS_CHARPATH   = np.zeros(num_sparsity)
EMP_RS_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
EMP_RS_BTW_CENTRALITY = np.zeros(num_sparsity)
EMP_RS_PARTICIPATION = np.zeros(num_sparsity)
EMP_RS_SMALL_WORLDNESS = np.zeros(num_sparsity)
EMP_RS_MODULARITY = np.zeros(num_sparsity)
#EMP_RS_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
EMP_RS_BW_RATIO   = np.zeros(num_sparsity)

TVB_RS_EFFICIENCY_G  = np.zeros(num_sparsity)
TVB_RS_EFFICIENCY_L = np.zeros(num_sparsity)
TVB_RS_CLUSTERING = np.zeros(num_sparsity)
TVB_RS_CHARPATH   = np.zeros(num_sparsity)
TVB_RS_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
TVB_RS_BTW_CENTRALITY = np.zeros(num_sparsity)
TVB_RS_PARTICIPATION = np.zeros(num_sparsity)
TVB_RS_SMALL_WORLDNESS = np.zeros(num_sparsity)
TVB_RS_MODULARITY = np.zeros(num_sparsity)
#TVB_RS_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
TVB_RS_BW_RATIO   = np.zeros(num_sparsity)

TVB_LSNM_RS_EFFICIENCY_G = np.zeros(num_sparsity)
TVB_LSNM_RS_EFFICIENCY_L = np.zeros(num_sparsity)
TVB_LSNM_RS_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_RS_CHARPATH   = np.zeros(num_sparsity)
TVB_LSNM_RS_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
TVB_LSNM_RS_BTW_CENTRALITY = np.zeros(num_sparsity)
TVB_LSNM_RS_PARTICIPATION = np.zeros(num_sparsity)
TVB_LSNM_RS_SMALL_WORLDNESS = np.zeros(num_sparsity)
TVB_LSNM_RS_MODULARITY = np.zeros(num_sparsity)
#TVB_LSNM_RS_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
TVB_LSNM_RS_BW_RATIO  = np.zeros(num_sparsity)

TVB_LSNM_PV_EFFICIENCY_G = np.zeros(num_sparsity)
TVB_LSNM_PV_EFFICIENCY_L = np.zeros(num_sparsity)
TVB_LSNM_PV_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_PV_CHARPATH   = np.zeros(num_sparsity)
TVB_LSNM_PV_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
TVB_LSNM_PV_BTW_CENTRALITY = np.zeros(num_sparsity)
TVB_LSNM_PV_PARTICIPATION = np.zeros(num_sparsity)
TVB_LSNM_PV_SMALL_WORLDNESS = np.zeros(num_sparsity)
TVB_LSNM_PV_MODULARITY = np.zeros(num_sparsity)
#TVB_LSNM_PV_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
TVB_LSNM_PV_BW_RATIO  = np.zeros(num_sparsity)

TVB_LSNM_DMS_EFFICIENCY_G = np.zeros(num_sparsity)
TVB_LSNM_DMS_EFFICIENCY_L = np.zeros(num_sparsity)
TVB_LSNM_DMS_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_DMS_CHARPATH   = np.zeros(num_sparsity)
TVB_LSNM_DMS_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
TVB_LSNM_DMS_BTW_CENTRALITY = np.zeros(num_sparsity)
TVB_LSNM_DMS_PARTICIPATION = np.zeros(num_sparsity)
TVB_LSNM_DMS_SMALL_WORLDNESS = np.zeros(num_sparsity)
TVB_LSNM_DMS_MODULARITY = np.zeros(num_sparsity)
#TVB_LSNM_DMS_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
TVB_LSNM_DMS_BW_RATIO  = np.zeros(num_sparsity)

#DMS_RS_DEGREE_DIFF = np.zeros((num_sparsity, lores_ROIs))

#RAND_MAT_EFFICIENCY=np.zeros(num_sparsity)
#RAND_MAT_CLUSTERING=np.zeros(num_sparsity)
#RAND_MAT_MODULARITY=np.zeros(num_sparsity)

# declare arrays to store binarized verions of FC matrices for all densities
emp_rs_bin = np.zeros((num_of_densities, lores_ROIs, lores_ROIs))
tvb_rs_bin = np.zeros((num_of_densities, lores_ROIs, lores_ROIs))
tvb_lsnm_rs_bin = np.zeros((num_of_densities, lores_ROIs, lores_ROIs))
tvb_lsnm_pv_bin = np.zeros((num_of_densities, lores_ROIs, lores_ROIs))
tvb_lsnm_dms_bin= np.zeros((num_of_densities, lores_ROIs, lores_ROIs))

emp_rs_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_rs_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_lsnm_rs_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_lsnm_pv_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_lsnm_dms_hr_bin= np.zeros((num_of_densities, hires_ROIs, hires_ROIs))


# generate a number of random weighted undirected FC matrices that will be used
# as a reference to compare functional connectivity matrices
#rand_mat = np.random.rand(num_random, lores_ROIs, lores_ROIs)
#for r in range(0, num_random):
#    rand_mat[r] = (rand_mat[r] + rand_mat[r].T) / 2       # make it symmetrical
#    np.fill_diagonal(rand_mat[r], 1.0)                    # fill diagonal with 1's


threshold_array = np.linspace(min_sparsity, max_sparsity, num_sparsity)

print '\nCalculating Graph metrics...'

for d in range(num_of_densities): 

    print 'Calculating graph metrics at density ', threshold_array[d]

    # calculate metrics for random matrices first. We are only interested in keeping an
    # average of each metric per random matrix.
#    for r in range(0, num_random):
#        rand_mat_p  = bct.threshold_proportional(rand_mat[r], threshold_array[d], copy=True)
#        rand_mat_bin= bct.binarize(rand_mat_p, copy=True)
#        rand_mat_global_efficiency = bct.efficiency_bin(rand_mat_bin, local=False)
#        RAND_MAT_EFFICIENCY[d] += rand_mat_global_efficiency
#        rand_mat_mean_clustering = np.mean(bct.clustering_coef_bu(rand_mat_bin))
#        RAND_MAT_CLUSTERING[d] += rand_mat_mean_clustering
#        rand_mat_modularity = bct.modularity_und(rand_mat_bin, gamma=1, kci=None)[1]
#        RAND_MAT_MODULARITY[d] += rand_mat_modularity
    # average graph metrics at the ith sparsity level across all random matrices
#    RAND_MAT_EFFICIENCY[d] = RAND_MAT_EFFICIENCY[d] / num_random
#    RAND_MAT_CLUSTERING[d] = RAND_MAT_CLUSTERING[d] / num_random
#    RAND_MAT_MODULARITY[d] = RAND_MAT_MODULARITY[d] / num_random

    # preprocess the functional connectivity matrices for each condition, for the current
    # subject and for the current density threshold. The preprocessing consists on thresholding
    # each fc matrix, then binarizing them.
    # Do these calculations for both, hi and low resolution FC matrices
    emp_rs_a           = np.absolute(empirical_fc_lowres)
    emp_rs_p           = bct.threshold_proportional(emp_rs_a, threshold_array[d], copy=True)
    emp_rs_bin[d]      = bct.binarize(emp_rs_p, copy=True)    
    
    emp_rs_a           = np.absolute(empirical_fc_hires)
    emp_rs_p           = bct.threshold_proportional(emp_rs_a, threshold_array[d], copy=True)
    emp_rs_hr_bin[d]   = bct.binarize(emp_rs_p, copy=True)

    tvb_rs_a           = np.absolute(tvb_rs_lowres_mean)
    tvb_rs_p           = bct.threshold_proportional(tvb_rs_a, threshold_array[d], copy=True)
    tvb_rs_bin[d]      = bct.binarize(tvb_rs_p, copy=True)

    tvb_rs_a           = np.absolute(tvb_rs_mean)
    tvb_rs_p           = bct.threshold_proportional(tvb_rs_a, threshold_array[d], copy=True)
    tvb_rs_hr_bin[d]   = bct.binarize(tvb_rs_p, copy=True)
    
    tvb_lsnm_rs_a      = np.absolute(tvb_lsnm_lowres_rs_mean)
    tvb_lsnm_rs_p      = bct.threshold_proportional(tvb_lsnm_rs_a, threshold_array[d], copy=True)
    tvb_lsnm_rs_bin[d] = bct.binarize(tvb_lsnm_rs_p, copy=True)

    tvb_lsnm_rs_a      = np.absolute(tvb_lsnm_rs_mean)
    tvb_lsnm_rs_p      = bct.threshold_proportional(tvb_lsnm_rs_a, threshold_array[d], copy=True)
    tvb_lsnm_rs_hr_bin[d]= bct.binarize(tvb_lsnm_rs_p, copy=True)

    tvb_lsnm_pv_a      = np.absolute(tvb_lsnm_lowres_pv_mean)
    tvb_lsnm_pv_p      = bct.threshold_proportional(tvb_lsnm_pv_a, threshold_array[d], copy=True)
    tvb_lsnm_pv_bin[d] = bct.binarize(tvb_lsnm_pv_p, copy=True)

    tvb_lsnm_pv_a      = np.absolute(tvb_lsnm_pv_mean)
    tvb_lsnm_pv_p      = bct.threshold_proportional(tvb_lsnm_pv_a, threshold_array[d], copy=True)
    tvb_lsnm_pv_hr_bin[d]= bct.binarize(tvb_lsnm_pv_p, copy=True)
    
    tvb_lsnm_dms_a      = np.absolute(tvb_lsnm_lowres_dms_mean)
    tvb_lsnm_dms_p      = bct.threshold_proportional(tvb_lsnm_dms_a, threshold_array[d], copy=True)
    tvb_lsnm_dms_bin[d] = bct.binarize(tvb_lsnm_dms_p, copy=True)

    tvb_lsnm_dms_a      = np.absolute(tvb_lsnm_dms_mean)
    tvb_lsnm_dms_p      = bct.threshold_proportional(tvb_lsnm_dms_a, threshold_array[d], copy=True)
    tvb_lsnm_dms_hr_bin[d] = bct.binarize(tvb_lsnm_dms_p, copy=True)

    # calculate global efficiency for each condition using Brain Connectivity Toolbox
#    EMP_RS_EFFICIENCY_G[d]      = bct.efficiency_bin(emp_rs_bin[d])  
#    TVB_RS_EFFICIENCY_G[d]      = bct.efficiency_bin(tvb_rs_bin[d]) 
    TVB_LSNM_RS_EFFICIENCY_G[d] = bct.efficiency_bin(tvb_lsnm_rs_bin[d]) 
    TVB_LSNM_PV_EFFICIENCY_G[d] = bct.efficiency_bin(tvb_lsnm_pv_bin[d])
    TVB_LSNM_DMS_EFFICIENCY_G[d]= bct.efficiency_bin(tvb_lsnm_dms_bin[d]) 

    # calculate mean local efficiency for each condition using Brain Connectivity Toolbox
#    EMP_RS_EFFICIENCY_L[d]      = np.mean(bct.efficiency_bin(emp_rs_bin[d], local=True))  
#    TVB_RS_EFFICIENCY_L[d]      = np.mean(bct.efficiency_bin(tvb_rs_bin[d], local=True)) 
#    TVB_LSNM_RS_EFFICIENCY_L[d] = np.mean(bct.efficiency_bin(tvb_lsnm_rs_bin[d], local=True)) 
#    TVB_LSNM_PV_EFFICIENCY_L[d] = np.mean(bct.efficiency_bin(tvb_lsnm_pv_bin[d], local=True))
#    TVB_LSNM_DMS_EFFICIENCY_L[d]= np.mean(bct.efficiency_bin(tvb_lsnm_dms_bin[d], local=True)) 
        
    # calculate mean clustering coefficient using Brain Connectivity Toolbox
#    EMP_RS_CLUSTERING[d]      = np.mean(bct.clustering_coef_bu(emp_rs_bin[d]))
#    TVB_RS_CLUSTERING[d]      = np.mean(bct.clustering_coef_bu(tvb_rs_bin[d]))
    TVB_LSNM_RS_CLUSTERING[d] = np.mean(bct.clustering_coef_bu(tvb_lsnm_rs_bin[d]))
    TVB_LSNM_PV_CLUSTERING[d] = np.mean(bct.clustering_coef_bu(tvb_lsnm_pv_bin[d]))
    TVB_LSNM_DMS_CLUSTERING[d] = np.mean(bct.clustering_coef_bu(tvb_lsnm_dms_bin[d]))

    # calculate characteristic path length using Brain Connectivity Toolbox. Please
    # note that you have to first calculate a distance matrix using distance_bin
#    emp_rs_bin_dist       = bct.distance_bin(emp_rs_bin[d])
#    tvb_rs_bin_dist       = bct.distance_bin(tvb_rs_bin[d])
#    tvb_lsnm_rs_bin_dist  = bct.distance_bin(tvb_lsnm_rs_bin[d])
#    tvb_lsnm_pv_bin_dist  = bct.distance_bin(tvb_lsnm_pv_bin[d])
#    tvb_lsnm_dms_bin_dist = bct.distance_bin(tvb_lsnm_dms_bin[d])
#    EMP_RS_CHARPATH[d]       = 1. / EMP_RS_EFFICIENCY_G[d]              #bct.charpath(emp_rs_bin_dist)[0]
#    TVB_RS_CHARPATH[d]       = 1. / TVB_RS_EFFICIENCY_G[d]              #bct.charpath(tvb_rs_bin_dist)[0]
#    TVB_LSNM_RS_CHARPATH[d]  = 1. / TVB_LSNM_RS_EFFICIENCY_G[d]         #bct.charpath(tvb_lsnm_rs_bin_dist)[0]
#    TVB_LSNM_PV_CHARPATH[d]  = 1. / TVB_LSNM_PV_EFFICIENCY_G[d]         #bct.charpath(tvb_lsnm_pv_bin_dist)[0]
#    TVB_LSNM_DMS_CHARPATH[d] = 1. / TVB_LSNM_DMS_EFFICIENCY_G[d]        #bct.charpath(tvb_lsnm_dms_bin_dist)[0]
    
    # calculate average eigenvector centrality using Brain Connectivity Toolbox
#    EMP_RS_EIGEN_CENTRALITY[d]      = np.mean(bct.eigenvector_centrality_und(emp_rs_bin[d]))
#    TVB_RS_EIGEN_CENTRALITY[d]      = np.mean(bct.eigenvector_centrality_und(tvb_rs_bin[d]))
#    TVB_LSNM_RS_EIGEN_CENTRALITY[d] = np.mean(bct.eigenvector_centrality_und(tvb_lsnm_rs_bin[d]))
#    TVB_LSNM_PV_EIGEN_CENTRALITY[d] = np.mean(bct.eigenvector_centrality_und(tvb_lsnm_pv_bin[d]))
#    TVB_LSNM_DMS_EIGEN_CENTRALITY[d] = np.mean(bct.eigenvector_centrality_und(tvb_lsnm_dms_bin[d]))

    # calculate average betweennes centrality using Brain Connectivity Toolbox
    # Note that we normalize the betweennes centrality prior to averaging
#    EMP_RS_BTW_CENTRALITY[d]      = np.mean(bct.betweenness_bin(emp_rs_bin[d]))/((lores_ROIs-1.0)*(lores_ROIs-2.0))
#    TVB_RS_BTW_CENTRALITY[d]      = np.mean(bct.betweenness_bin(tvb_rs_bin[d]))/((lores_ROIs-1.0)*(lores_ROIs-2.0))
#    TVB_LSNM_RS_BTW_CENTRALITY[d] = np.mean(bct.betweenness_bin(tvb_lsnm_rs_bin[d]))/((lores_ROIs-1.0)*(lores_ROIs-2.0))
#    TVB_LSNM_PV_BTW_CENTRALITY[d] = np.mean(bct.betweenness_bin(tvb_lsnm_pv_bin[d]))/((lores_ROIs-1.0)*(lores_ROIs-2.0))
#    TVB_LSNM_DMS_BTW_CENTRALITY[d] = np.mean(bct.betweenness_bin(tvb_lsnm_dms_bin[d]))/((lores_ROIs-1.0)*(lores_ROIs-2.0))

    # calculate average participation coefficient using Brain Connectivity Toolbox
#    EMP_RS_PARTICIPATION[d]      = np.mean(bct.participation_coef(emp_rs_bin[d], CI, degree='undirected'))
#    TVB_RS_PARTICIPATION[d]      = np.mean(bct.participation_coef(tvb_rs_bin[d], CI, degree='undirected'))
#    TVB_LSNM_RS_PARTICIPATION[d] = np.mean(bct.participation_coef(tvb_lsnm_rs_bin[d], CI, degree='undirected'))
#    TVB_LSNM_PV_PARTICIPATION[d] = np.mean(bct.participation_coef(tvb_lsnm_pv_bin[d], CI, degree='undirected'))
#    TVB_LSNM_DMS_PARTICIPATION[d]= np.mean(bct.participation_coef(tvb_lsnm_dms_bin[d], CI, degree='undirected'))


    
    # calculate modularity using Brain Connectivity Toolbox
#    emp_modularity             = bct.modularity_und(emp_rs_bin[d], gamma=1, kci=None)
#    tvb_rs_modularity          = bct.modularity_und(tvb_rs_bin[d], gamma=1, kci=None)
    tvb_lsnm_rs_modularity     = bct.modularity_und(tvb_lsnm_rs_bin[d], gamma=1, kci=None)
    tvb_lsnm_pv_modularity     = bct.modularity_und(tvb_lsnm_pv_bin[d], gamma=1, kci=None)
    tvb_lsnm_dms_modularity    = bct.modularity_und(tvb_lsnm_dms_bin[d], gamma=1, kci=None)
#    EMP_RS_MODULARITY[d]       = emp_modularity[1]
#    TVB_RS_MODULARITY[d]       = tvb_rs_modularity[1]
    TVB_LSNM_RS_MODULARITY[d]  = tvb_lsnm_rs_modularity[1]
    TVB_LSNM_PV_MODULARITY[d]  = tvb_lsnm_pv_modularity[1]
    TVB_LSNM_DMS_MODULARITY[d] = tvb_lsnm_dms_modularity[1]

    # print the modules resulting from the modularity calculation above
#    print "Empirical RS modules: ", emp_modularity[0]
#    print "TVB RS modules: ", tvb_rs_modularity[0]
#    print "TVB/LSNM RS modules: ", tvb_lsnm_rs_modularity[0]
#    print "TVB/LSNM PV modules: ", tvb_lsnm_pv_modularity[0]
#    print "TVB/LSNM DMS modules: ", tvb_lsnm_dms_modularity[0]
        
    # calculate node degree using BCT
#    EMP_RS_DEGREE[d]          = bct.degrees_und(emp_rs_bin[d])
#    TVB_RS_DEGREE[d]          = bct.degrees_und(tvb_rs_bin[d])
#    TVB_LSNM_RS_DEGREE[d]     = bct.degrees_und(tvb_lsnm_rs_bin[d])
#    TVB_LSNM_PV_DEGREE[d]     = bct.degrees_und(tvb_lsnm_pv_bin[d])
#    TVB_LSNM_DMS_DEGREE[d]    = bct.degrees_und(tvb_lsnm_dms_bin[d])

    # calculate the node degree differences at each density treshold
#    DMS_RS_DEGREE_DIFF[d] = TVB_LSNM_DMS_DEGREE[d] - TVB_LSNM_RS_DEGREE[d]
    
    # calculate number of between-module connections...
    # and the number of within-module connections...
    # and use those to calculate the ratio of the avg between- to avg within-module connections
    # Note that we are using hi-res binarized FC matrices for this metric as each one of the hires
    # nodes belongs to one lo-res module
#    between_mod_cxn = np.zeros(5)    # prepare empty buckets for between-module connections for 4 conditions + emp
#    within_mod_cxn  = np.zeros(5)    # prepare empty buckets for within-module connections for 4 conditions + emp
#    for a in range(0, hires_ROIs):
#        for b in range(0, hires_ROIs):
#            x = hagmann_empirical['roi_lbls'][0][a]
#            y = hagmann_empirical['roi_lbls'][0][b]
#            if tvb_rs_hr_bin[d, a, b] == 1:
#                    
#                if x == y:
#                    within_mod_cxn[0]  += 1
#                else :
#                    between_mod_cxn[0] +=1
#
#            if tvb_lsnm_rs_hr_bin[d, a, b] == 1:
#
#                if x == y:
#                    within_mod_cxn[1]  +=1
#                else:
#                    between_mod_cxn[1] +=1
#
#            if tvb_lsnm_pv_hr_bin[d, a, b] == 1:
#
#                if x == y:
#                    within_mod_cxn[2]  +=1
#                else:
#                    between_mod_cxn[2] +=1
#
#            if tvb_lsnm_dms_hr_bin[d, a, b] == 1:
#
#                if x == y:
#                    within_mod_cxn[3]  +=1
#                else:
#                    between_mod_cxn[3] +=1
#            if emp_rs_hr_bin[d, a, b] == 1:
#
#                if x == y:
#                    within_mod_cxn[4] +=1
#                else:
#                    between_mod_cxn[4] +=1
#                        
#    # calculate max potential number of within-module connections for all modules
#    max_wth_cxn = 0
#    for a in range(0, lores_ROIs):
#        max_wth_cxn += freq_array[a][1] * freq_array[a][1]
#    max_btw_cxn = hires_ROIs * hires_ROIs - max_wth_cxn
#
#    avg_btw_cxn = np.zeros(4)
#    avg_wth_cxn = np.zeros(4)
#    avg_wth_cxn = within_mod_cxn.astype(float)  / float(max_wth_cxn)
#    avg_btw_cxn = between_mod_cxn.astype(float) / float(max_btw_cxn)
#    # calculate the ratio of B- to W- module connections    
#    TVB_RS_BW_RATIO[d]        = avg_btw_cxn[0] / avg_wth_cxn[0]
#    TVB_LSNM_RS_BW_RATIO[d]   = avg_btw_cxn[1] / avg_wth_cxn[1]
#    TVB_LSNM_PV_BW_RATIO[d]   = avg_btw_cxn[2] / avg_wth_cxn[2]
#    TVB_LSNM_DMS_BW_RATIO[d]  = avg_btw_cxn[3] / avg_wth_cxn[3]
#    EMP_RS_BW_RATIO[d]        = avg_btw_cxn[4] / avg_wth_cxn[4]
#    
    # save unweighted undirected matrix at the given sparsity level
    # then, convert the matrix to an edge table and save that as well.
    #if p == 0.2:
    #    df = pd.DataFrame(tvb_rs_mean_bin, index=labels, columns=labels)
    #    df.to_csv('tvb_rs_mean_bin.csv')
    #    adj_to_list('tvb_rs_mean_bin.csv', 'tvb_rs_mean_edge_table.csv', ',')
    
# generate a number of random matrices corresponding to each density threshold, to be used
# to normalize network metrics. The randomized matrices have same number of nodes and edges
# as the corresponding matrices in each density threshold.
#rnd_clustering = np.zeros((num_random, num_of_densities))
#rnd_eff_G   = np.zeros((num_random, num_of_densities))

print '\n Finished calculating graph metrics...'

#rand_mat = np.random.rand(num_random, hires_ROIs, hires_ROIs)
#for r in range(0, num_random):
#
#    rand_symm = (rand_mat[r] + rand_mat[r].T) / 2         # make it symmetrical
#    np.fill_diagonal(rand_symm, 1.0)                      # fill diagonal with 1's
#
#    for d in range(0, num_of_densities):
#
#        rand_p=bct.threshold_proportional(rand_symm, threshold_array[d], copy=True) # threshold random matrix
#        rand_bin=bct.binarize(rand_p, copy=True)                                    # binarize random matrix
#        
#        rnd_clustering[r, d] = np.mean(bct.clustering_coef_bu(rand_bin))
#        rnd_eff_G[r, d] = bct.efficiency_bin(rand_bin)
#
# calculate the average network metric for each density threshold across all random matrices
#clustering_mean = np.mean(rnd_clustering, axis=0)
#eff_G_mean_R = np.mean(rnd_eff_G, axis=0)
# normalize the network metrics that depend on density
# number of nodes and connections and same node distribution
#for d in range(0, num_of_densities):
#
    # calculate small-worldness (ratio between normalized clustering coefficient and
    # normalized characteristic path length)
#    EMP_RS_SMALL_WORLDNESS[d] = (EMP_RS_CLUSTERING[d] /clustering_mean[d]) / (eff_G_mean_R[d] / EMP_RS_EFFICIENCY_G[d])
#    TVB_RS_SMALL_WORLDNESS[d] = (TVB_RS_CLUSTERING[d] /clustering_mean[d]) / (eff_G_mean_R[d] / TVB_RS_EFFICIENCY_G[d]) 
#    TVB_LSNM_RS_SMALL_WORLDNESS[d] = (TVB_LSNM_RS_CLUSTERING[d] /clustering_mean[d]) / (eff_G_mean_R[d] / TVB_LSNM_RS_EFFICIENCY_G[d])
#    TVB_LSNM_PV_SMALL_WORLDNESS[d] = (TVB_LSNM_PV_CLUSTERING[d] /clustering_mean[d]) / (eff_G_mean_R[d] / TVB_LSNM_PV_EFFICIENCY_G[d])
#    TVB_LSNM_DMS_SMALL_WORLDNESS[d] = (TVB_LSNM_DMS_CLUSTERING[d] /clustering_mean[d]) / (eff_G_mean_R[d] / TVB_LSNM_DMS_EFFICIENCY_G[d])
#
#    TVB_LSNM_RS_EFFICIENCY_G[d] = TVB_LSNM_RS_EFFICIENCY_G[d] / eff_G_mean_R[d]
#    TVB_LSNM_PV_EFFICIENCY_G[d] = TVB_LSNM_PV_EFFICIENCY_G[d] / eff_G_mean_R[d]
#    TVB_LSNM_DMS_EFFICIENCY_G[d] = TVB_LSNM_DMS_EFFICIENCY_G[d] / eff_G_mean_R[d]

    
# calculate nodal degree using BCT
#tvb_rs_nodal_degree = bct.degrees_und(tvb_rs_p)

# calculate nodal strength using BCT
#tvb_rs_nodal_strength = bct.strengths_und_sign(tvb_rs_p)

# calculate nodal degree using BCT
#tvb_lsnm_rs_nodal_degree = bct.degrees_und(tvb_lsnm_rs_p)

# calculate nodal strength using BCT
#tvb_lsnm_rs_nodal_strength = bct.strengths_und_sign(tvb_lsnm_rs_p)

# calculate nodal degree using BCT
#tvb_lsnm_pv_nodal_degree = bct.degrees_und(tvb_lsnm_pv_p)

# calculate nodal strength using BCT
#tvb_lsnm_pv_nodal_strength = bct.strengths_und_sign(tvb_lsnm_pv_p)

# calculate nodal degree using BCT
#tvb_lsnm_dms_nodal_degree = bct.degrees_und(tvb_lsnm_dms_p)

# calculate nodal strength using BCT
#tvb_lsnm_dms_nodal_strength = bct.strengths_und_sign(tvb_lsnm_dms_p)

###############################################################################
# Display heatmaps of all low-res FC matrices, weighted and binarized,
# empirical and simulated for all conditions
###############################################################################
#fig = plt.figure('Mean TVB-only RS FC')
#ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
#cmap = CM.get_cmap('jet', 10)
#cax = ax.imshow(tvb_rs_lowres_mean, vmin=-0.5, vmax=0.5, interpolation='nearest', cmap='bwr')
#ax.grid(False)
#color_bar=plt.colorbar(cax)
#fig.savefig('mean_tvb_only_rs_fc.png')

fig = plt.figure('Mean TVB/LSNM PF FC')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_lowres_rs_mean,
                #vmin=-0.53, vmax=0.53,
                interpolation='nearest', cmap='bwr')
ax.grid(False)
color_bar=plt.colorbar(cax)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('mean_tvb_lsnm_rs_fc_incl_PreSMA_Fixation.png')

fig = plt.figure('Mean TVB/LSNM PV FC')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_lowres_pv_mean,
                #vmin=-0.53, vmax=0.53,
                interpolation='nearest', cmap='bwr')
ax.grid(False)
color_bar=plt.colorbar(cax)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('mean_tvb_lsnm_pv_fc_incl_PreSMA_Fixation.png')

fig = plt.figure('Mean TVB/LSNM DMS FC')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_lowres_dms_mean,
                #vmin=-0.53, vmax=0.53,
                interpolation='nearest', cmap='bwr')
ax.grid(False)
color_bar=plt.colorbar(cax)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('mean_tvb_lsnm_dms_fc_incl_PreSMA_Fixation.png')

#fig=plt.figure('Functional Connectivity Matrix of empirical BOLD (66 ROIs)')
#ax = fig.add_subplot(111)
#empirical_fc_hires = np.asarray(empirical_fc_hires)
#cax = ax.imshow(empirical_fc_lowres, vmin=-0.5, vmax=0.5, interpolation='nearest', cmap='bwr')
#ax.grid(False)
#color_bar=plt.colorbar(cax)
#fig.savefig('empirical_rs_fc.png')

#fig = plt.figure('Mean TVB-only RS FC (binary at 40% sparsity)')
#ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
#cax = ax.imshow(tvb_rs_bin[-1], interpolation='nearest', cmap='Greys')
#ax.grid(False)
#fig.savefig('binary_tvb_only_rs_fc.png')

fig = plt.figure('Mean TVB/LSNM RS FC (binarized at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_rs_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('binary_tvb_lsnm_rs_fc_incl_PreSMA_Fixation.png')

fig = plt.figure('Mean TVB/LSNM PV FC (binarized at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_pv_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('binary_tvb_lsnm_pv_fc_incl_PreSMA_Fixation.png')

fig = plt.figure('Mean TVB/LSNM DMS FC (binarized at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_dms_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('binary_tvb_lsnm_dms_fc_incl_PreSMA_Fixation.png')

#fig = plt.figure('Empirical RS FC matrix (binarized at 40% sparsity)')
#ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
#cax = ax.imshow(emp_rs_bin[-1], interpolation='nearest', cmap='Greys')
#ax.grid(False)
#fig.savefig('binary_empirical_fc.png')

####################################################################################
# Display heatmaps showing differences between PV and RS, DMS and RS
####################################################################################
#fig = plt.figure('TVB/LSNM PV - TVB/LSNM RS')
#ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
#cax = ax.imshow(pv_minus_rs, vmin=FC_min, vmax=FC_max, interpolation='nearest', cmap='OrRd')
#cax = ax.imshow(pv_minus_rs, vmin=-0.26, vmax=0.26, interpolation='nearest', cmap='bwr')
#ax.grid(False)
#color_bar=plt.colorbar(cax)
#fig.savefig('pv_minus_rs.png')
#
# change frequency of ticks to match number of ROI labels
#plt.xticks(np.arange(0, len(labels)))
#plt.yticks(np.arange(0, len(labels)))
#
# decrease font size
#plt.rcParams.update({'font.size': 9})
#
# display labels for brain regions
#ax.set_xticklabels(labels, rotation=90)
#ax.set_yticklabels(labels)
#
# Turn off all the ticks
#ax = plt.gca()
#
#for t in ax.xaxis.get_major_ticks():
#    t.tick1On = False
#    t.tick2On = False
#for t in ax.yaxis.get_major_ticks():
#    t.tick1On = False
#    t.tick2On = False
#
#fig = plt.figure('TVB/LSNM DMS - TVB/LSNM RS')
#ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
#cax = ax.imshow(dms_minus_rs, vmin=FC_min, vmax=FC_max, interpolation='nearest', cmap='OrRd')
#cax = ax.imshow(dms_minus_rs, vmin=-0.26, vmax=0.26, interpolation='nearest', cmap='bwr')
#ax.grid(False)
#color_bar=plt.colorbar(cax)
#fig.savefig('dms_minus_rs.png')
#
# change frequency of ticks to match number of ROI labels
#plt.xticks(np.arange(0, len(labels)))
#plt.yticks(np.arange(0, len(labels)))
#
# display labels for brain regions
#ax.set_xticklabels(labels, rotation=90)
#ax.set_yticklabels(labels)
#
# Turn off all the ticks
#ax = plt.gca()
#
#for t in ax.xaxis.get_major_ticks():
#    t.tick1On = False
#    t.tick2On = False
#for t in ax.yaxis.get_major_ticks():
#    t.tick1On = False
#    t.tick2On = False

#################################################################################
# Plot correlation coefficients for using designated seeds found in RS literature
#################################################################################
# plot the correlation coefficients for empirical rPC
#fig = plt.figure('Empirical correlation coefficients using rPC as seed')
#ax = fig.add_subplot(111)
#rPC_loc   = labels.index('  rPC')
#y_pos = np.arange(len(labels))
#ax.barh(y_pos, empirical_fc_lowres[rPC_loc])
#ax.set_yticks(y_pos)
#ax.set_yticklabels(labels)
#ax.invert_yaxis()
#ax.set_xlabel('Correlation coefficient')

# plot the correlation coefficients for empirical lPCUN
#fig = plt.figure('Empirical correlation coefficients using lPCUN as seed')
#ax = fig.add_subplot(111)
#lPCUN_loc   = labels.index('lPCUN')
#y_pos = np.arange(len(labels))
#ax.barh(y_pos, empirical_fc_lowres[lPCUN_loc])
#ax.set_yticks(y_pos)
#ax.set_yticklabels(labels)
#ax.invert_yaxis()
#ax.set_xlabel('Correlation coefficient')


# plot the correlation coefficients for simulated rPC
#fig = plt.figure('Model correlation coefficients using rPC as seed')
#ax = fig.add_subplot(111)
#rPC_loc   = labels.index('  rPC')
#y_pos = np.arange(len(labels))
#ax.barh(y_pos, tvb_lsnm_lowres_rs_mean[rPC_loc])
#ax.set_yticks(y_pos)
#ax.set_yticklabels(labels)
#ax.invert_yaxis()
#ax.set_xlabel('Correlation coefficient')

# plot the correlation coefficients for simulated lPCUN
#fig = plt.figure('Model correlation coefficients using lPCUN as seed')
#ax = fig.add_subplot(111)
#lPCUN_loc   = labels.index('lPCUN')
#y_pos = np.arange(len(labels))
#ax.barh(y_pos, tvb_rs_lowres_mean[lPCUN_loc])
#ax.set_yticks(y_pos)
#ax.set_yticklabels(labels)
#ax.invert_yaxis()
#ax.set_xlabel('Correlation coefficient')

####################################################################################
# Display graph theory metrics of RS empirical vs simulated all sparsity thresholds
####################################################################################
#fig = plt.figure('Global Efficiency during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_EFFICIENCY_G, label='Empirical')
#plt.plot(threshold_array, TVB_RS_EFFICIENCY_G, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_EFFICIENCY_G, label='TVB/LSNM RS')
#plt.plot(threshold_array, RAND_MAT_EFFICIENCY, label='Random')
#plt.xlabel('Threshold')
#plt.ylabel('Mean Global Efficiency')
#plt.legend(loc='best')
#fig.savefig('emp_sim_global_efficiency_across_thresholds.png')

#fig = plt.figure('Mean Local Efficiency during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_EFFICIENCY_L, label='Empirical')
#plt.plot(threshold_array, TVB_RS_EFFICIENCY_L, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_EFFICIENCY_L, label='TVB/LSNM RS')
#plt.xlabel('Density threshold')
#plt.ylabel('Mean Local Efficiency')
#plt.legend(loc='best')
#fig.savefig('emp_sim_mean_local_efficiencies_across_densities.png')

#fig = plt.figure('Mean Clustering Coefficient during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_CLUSTERING, label='Empirical')
#plt.plot(threshold_array, TVB_RS_CLUSTERING, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_CLUSTERING, label='TVB/LSNM RS')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
#plt.xlabel('Threshold')
#plt.ylabel('Mean Clustering')
#plt.legend(loc='best')
#fig.savefig('emp_sim_clustering_across_thresholds.png')

#fig = plt.figure('Modularity during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_MODULARITY, label='Empirical')
#plt.plot(threshold_array, TVB_RS_MODULARITY, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_MODULARITY, label='TVB/LSNM RS')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
#plt.xlabel('Density threshold')
#plt.ylabel('Modularity')
#plt.legend(loc='best')
#fig.savefig('emp_sim_modularity_across_thresholds.png')

#fig = plt.figure('Characteristic path length during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_CHARPATH, label='Empirical')
#plt.plot(threshold_array, TVB_RS_CHARPATH, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_CHARPATH, label='TVB/LSNM RS')
#plt.xlabel('Threshold')
#plt.ylabel('Characteristic Path Length')
#plt.legend(loc='best')
#fig.savefig('emp_sim_charpath_across_thresholds.png')

#fig = plt.figure('Average Eigenvector Centrality during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_EIGEN_CENTRALITY, label='Empirical')
#plt.plot(threshold_array, TVB_RS_EIGEN_CENTRALITY, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_EIGEN_CENTRALITY, label='TVB/LSNM RS')
#plt.xlabel('Threshold')
#plt.ylabel('Average Eigenvector Centrality')
#plt.legend(loc='best')
#fig.savefig('emp_sim_eigen_centrality_across_thresholds.png')

#fig = plt.figure('Average Betweennes Centrality during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_BTW_CENTRALITY, label='Empirical')
#plt.plot(threshold_array, TVB_RS_BTW_CENTRALITY, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_BTW_CENTRALITY, label='TVB/LSNM RS')
#plt.xlabel('Threshold')
#plt.ylabel('Average Betweennes Centrality')
#plt.legend(loc='best')
#fig.savefig('emp_sim_btwn_centrality_across_thresholds.png')

#fig = plt.figure('Average Participation Coefficient during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_PARTICIPATION, label='Empirical')
#plt.plot(threshold_array, TVB_RS_PARTICIPATION, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_PARTICIPATION, label='TVB/LSNM RS')
#plt.xlabel('Threshold')
#plt.ylabel('Average Participation Coefficient')
#plt.legend(loc='best')
#fig.savefig('emp_sim_participation_across_thresholds.png')

#fig = plt.figure('Small Worldness during RS')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, EMP_RS_SMALL_WORLDNESS, label='Empirical')
#plt.plot(threshold_array, TVB_RS_SMALL_WORLDNESS, label='TVB RS')
#plt.plot(threshold_array, TVB_LSNM_RS_SMALL_WORLDNESS, label='TVB/LSNM RS')
#plt.xlabel('Threshold')
#plt.ylabel('Small Worldness')
#plt.legend(loc='best')
#fig.savefig('emp_sim_small_worldness_across_thresholds.png')

####################################################################################
# Display graph theory metrics of RS, PV, DMS for all sparsity thresholds
####################################################################################
fig = plt.figure('Global Efficiency')
cax = fig.add_subplot(111)
plt.plot(threshold_array, TVB_LSNM_RS_EFFICIENCY_G, label='TVB/LSNM PF')
plt.plot(threshold_array, TVB_LSNM_PV_EFFICIENCY_G, label='TVB/LSNM PV')
plt.plot(threshold_array, TVB_LSNM_DMS_EFFICIENCY_G, label='TVB/LSNM DMS')
#plt.plot(threshold_array, RAND_MAT_EFFICIENCY, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Mean Global Efficiency')
plt.legend(loc='best')
fig.savefig('rs_pv_dms_global_efficiency_across_thresholds.png')

#fig = plt.figure('Mean Local Efficiency')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_EFFICIENCY_L, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_EFFICIENCY_L, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_EFFICIENCY_L, label='TVB/LSNM DMS')
#plt.xlabel('Density threshold')
#plt.ylabel('Mean Local Efficiency')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_mean_local_efficiencies_across_densities.png')

fig = plt.figure('Mean Clustering Coefficient')
cax = fig.add_subplot(111)
plt.plot(threshold_array, TVB_LSNM_RS_CLUSTERING, label='TVB/LSNM PF')
plt.plot(threshold_array, TVB_LSNM_PV_CLUSTERING, label='TVB/LSNM PV')
plt.plot(threshold_array, TVB_LSNM_DMS_CLUSTERING, label='TVB/LSNM DMS')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Mean Clustering')
plt.legend(loc='best')
fig.savefig('rs_pv_dms_clustering_across_thresholds.png')

fig = plt.figure('Modularity')
cax = fig.add_subplot(111)
plt.plot(threshold_array, TVB_LSNM_RS_MODULARITY, label='TVB/LSNM PF')
plt.plot(threshold_array, TVB_LSNM_PV_MODULARITY, label='TVB/LSNM PV')
plt.plot(threshold_array, TVB_LSNM_DMS_MODULARITY, label='TVB/LSNM DMS')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
plt.xlabel('Density threshold')
plt.ylabel('Modularity')
plt.legend(loc='best')
fig.savefig('rs_pv_dms_modularity_across_thresholds.png')

#fig = plt.figure('B/W RATIO')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_BW_RATIO, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_BW_RATIO, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_BW_RATIO, label='TVB/LSNM DMS')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
#plt.xlabel('Density threshold')
#plt.ylabel('B/W RATIO')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_bw_ratio_across_thresholds.png')

#fig = plt.figure('Characteristic path length of a range of densities')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_CHARPATH, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_CHARPATH, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_CHARPATH, label='TVB/LSNM DMS')
#plt.xlabel('Threshold')
#plt.ylabel('Characteristic Path Length')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_charpath_across_thresholds.png')

#fig = plt.figure('Average Eigenvector Centrality for a range of densities')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_EIGEN_CENTRALITY, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_EIGEN_CENTRALITY, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_EIGEN_CENTRALITY, label='TVB/LSNM DMS')
#plt.xlabel('Threshold')
#plt.ylabel('Average Eigenvector Centrality')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_eigen_centrality_across_thresholds.png')

#fig = plt.figure('Average Betweennes Centrality for a range of densities')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_BTW_CENTRALITY, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_BTW_CENTRALITY, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_BTW_CENTRALITY, label='TVB/LSNM DMS')
#plt.xlabel('Threshold')
#plt.ylabel('Average Betweennes Centrality')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_btwn_centrality_across_thresholds.png')

#fig = plt.figure('Average Participation Coefficient')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_PARTICIPATION, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_PARTICIPATION, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_PARTICIPATION, label='TVB/LSNM DMS')
#plt.xlabel('Threshold')
#plt.ylabel('Average Participation Coefficient')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_participation_across_thresholds.png')

#fig = plt.figure('Small Worldness')
#cax = fig.add_subplot(111)
#plt.plot(threshold_array, TVB_LSNM_RS_SMALL_WORLDNESS, label='TVB/LSNM RS')
#plt.plot(threshold_array, TVB_LSNM_PV_SMALL_WORLDNESS, label='TVB/LSNM PV')
#plt.plot(threshold_array, TVB_LSNM_DMS_SMALL_WORLDNESS, label='TVB/LSNM DMS')
#plt.xlabel('Threshold')
#plt.ylabel('Small Worldness')
#plt.legend(loc='best')
#fig.savefig('rs_pv_dms_small_worldness_across_thresholds.png')


######################## TESTING 3D PLOT OF ADJACENCY MATRIX (NEEDS MAYAVI)

#white_matter = connectivity.Connectivity.from_file("connectivity_66.zip")
#coor = white_matter.centres
#fig=bct.adjacency_plot_und(tvb_lsnm_dms_mean_p,coor)
#mlab.show()

######################## TESTING 3D PLOT OF ADJACENCY MATRIX (NEEDS MAYAVI)

# the following plots bar graphs in subplots, one for each experimental condition,
# where each bar represents the nodal strength of each node
# Four subplots sharing both x/y axes

# initialize new figure for nodal strength bar charts
#index = np.arange(66)
#bar_width = 1
#colors = 'lightblue '*66            # create array of colors for bar chart
#c_tvb_rs_s = colors.split()         # one for each bar chart type
#c_tvb_lsnm_rs_s = colors.split()
#c_tvb_lsnm_pv_s = colors.split()
#c_tvb_lsnm_dms_s = colors.split()
#c_tvb_rs_k = colors.split()
#c_tvb_lsnm_rs_k = colors.split()
#c_tvb_lsnm_pv_k = colors.split()
#c_tvb_lsnm_dms_k = colors.split()

# Find 10 maximum values for each condition and for each metric,
# then highlight top 10 by changing bar color to red
#top_10_s1 = tvb_rs_nodal_strength[0].argsort()[-10:][::-1]
#for idx in top_10_s1:
#    c_tvb_rs_s[idx] = 'red'
#top_10_s2 = tvb_lsnm_rs_nodal_strength[0].argsort()[-10:][::-1]
#for idx in top_10_s2:
#    c_tvb_lsnm_rs_s[idx] = 'red'
#top_10_s3 = tvb_lsnm_pv_nodal_strength[0].argsort()[-10:][::-1]
#for idx in top_10_s3:
#    c_tvb_lsnm_pv_s[idx] = 'red'
#top_10_s4 = tvb_lsnm_dms_nodal_strength[0].argsort()[-10:][::-1]
#for idx in top_10_s4:
#    c_tvb_lsnm_dms_s[idx] = 'red'
#top_10_k1 = tvb_rs_nodal_degree.argsort()[-10:][::-1]
#for idx in top_10_k1:
#    c_tvb_rs_k[idx] = 'red'
#top_10_k2 = tvb_lsnm_rs_nodal_degree.argsort()[-10:][::-1]
#for idx in top_10_k2:
#    c_tvb_lsnm_rs_k[idx] = 'red'
#top_10_k3 = tvb_lsnm_pv_nodal_degree.argsort()[-10:][::-1]
#for idx in top_10_k3:
#    c_tvb_lsnm_pv_k[idx] = 'red'
#top_10_k4 = tvb_lsnm_dms_nodal_degree.argsort()[-10:][::-1]
#for idx in top_10_k4:
#    c_tvb_lsnm_dms_k[idx] = 'red'
    
# generate bar charts
#f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
#ax1.bar(index, tvb_rs_nodal_strength[0], bar_width, color=c_tvb_rs_s)
#ax1.set_title('Nodal strength for all 66 nodes')
#ax2.bar(index, tvb_lsnm_rs_nodal_strength[0], bar_width, color=c_tvb_lsnm_rs_s)
#ax3.bar(index, tvb_lsnm_pv_nodal_strength[0], bar_width, color=c_tvb_lsnm_pv_s)
#ax4.bar(index, tvb_lsnm_dms_nodal_strength[0], bar_width, color=c_tvb_lsnm_dms_s)

#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.xticks(index + bar_width/2.0, labels, rotation='vertical')

# initialize new figure for nodal degree bar charts
#f, (ax5, ax6, ax7, ax8) = plt.subplots(4, sharex=True, sharey=True)
#ax5.bar(index, tvb_rs_nodal_degree, bar_width, color=c_tvb_rs_k)
#ax5.set_title('Nodal degree for all 66 nodes')
#ax6.bar(index, tvb_lsnm_rs_nodal_degree, bar_width, color=c_tvb_lsnm_rs_k)
#ax7.bar(index, tvb_lsnm_pv_nodal_degree, bar_width, color=c_tvb_lsnm_pv_k)
#ax8.bar(index, tvb_lsnm_dms_nodal_degree, bar_width, color=c_tvb_lsnm_dms_k)

#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.xticks(index + bar_width/2.0, labels, rotation='vertical')


##### TESTING CIRCULAR GRAPHS FOR MAJOR FUNCTIONAL CONNECTIONS
#label_names = labels
#node_order = labels
#node_angles = circular_layout(label_names, node_order, start_pos=90,
#                              group_boundaries=[0, len(label_names) / 2])
#
# We only show the strongest connections.
#fig = plt.figure('TVB RS FC')
#fig.patch.set_facecolor('black')
#ax = fig.add_subplot(111)
#print tvb_rs_mean.shape
#plot_connectivity_circle(tvb_rs_mean, label_names, n_lines=30,
#                         node_angles=node_angles, 
#                         title='TVB Resting State Functional Connectivity',
#                         vmin=-1, vmax=1, colormap=cmap, fig=fig)
#fig.savefig('circle_tvb_rs_fc.png', facecolor='black')
# We only show the strongest connections.
#fig = plt.figure('TVB/LSNM RS FC')
#fig.patch.set_facecolor('black')
#ax = fig.add_subplot(111)
#plot_connectivity_circle(tvb_lsnm_rs_mean, label_names, n_lines=30,
#                         node_angles=node_angles, 
#                         title='TVB/LSNM Resting State Functional Connectivity',
#                         vmin=-1, vmax=1, colormap=cmap, fig=fig)
#fig.savefig('circle_tvb_lsnm_rs_fc.png', facecolor='black')
# We only show the strongest connections.
#fig = plt.figure('TVB/LSNM PV FC')
#fig.patch.set_facecolor('black')
#ax = fig.add_subplot(111)
#plot_connectivity_circle(tvb_lsnm_pv_mean, label_names, n_lines=30,
#                         node_angles=node_angles, 
#                         title='TVB/LSNM Passive Viewing Functional Connectivity',
#                         vmin=-1, vmax=1, colormap=cmap, fig=fig)
#fig.savefig('circle_tvb_lsnm_pv_fc.png', facecolor='black')
# We only show the strongest connections.
#fig = plt.figure('TVB/LSNM DMS FC')
#fig.patch.set_facecolor('black')
#ax = fig.add_subplot(111)
#plot_connectivity_circle(tvb_lsnm_dms_mean, label_names, n_lines=30,
#                         node_angles=node_angles, 
#                         title='TVB/LSNM DMS Task Functional Connectivity',
#                         vmin=-1, vmax=1, colormap=cmap, fig=fig)
#fig.savefig('circle_tvb_lsnm_dms_fc.png', facecolor='black')
##### TESTING A CIRCULAR GRAPH FOR MAJOR FUNCTIONAL CONNECTIONS


# initialize new figure for tvb resting-state histogram
#fig = plt.figure('TVB Resting State')
#cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
#mask = np.tri(tvb_rs_mean.shape[0], k=0)
#mask = np.transpose(mask)
#masked_tvb_rs_mean = np.ma.array(tvb_rs_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
#corr_mat_tvb_rs = np.ma.ravel(masked_tvb_rs_mean)

# remove masked elements from cross-correlation matrix
#corr_mat_tvb_rs = np.ma.compressed(corr_mat_tvb_rs)

# plot a histogram to show the frequency of correlations
#plt.hist(corr_mat_tvb_rs, 25)

#plt.xlabel('Correlation Coefficient')
#plt.ylabel('Number of occurrences')
#plt.axis([-1, 1, 0, 600])

#fig.savefig('tvb_rs_hist_66_ROIs')


# calculate and print kurtosis
#print '\nTVB Resting-State Fishers kurtosis: ', kurtosis(corr_mat_tvb_rs, fisher=True)
#print 'TVB Resting-State Skewness: ', skew(corr_mat_tvb_rs)

# initialize new figure for tvb/lsnm resting-state histogram
#fig = plt.figure('TVB/LSNM Resting State')
#cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
#mask = np.tri(tvb_lsnm_rs_mean.shape[0], k=0)
#mask = np.transpose(mask)
#tvb_lsnm_rs_mean = np.ma.array(tvb_lsnm_rs_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
#corr_mat_tvb_lsnm_rs = np.ma.ravel(tvb_lsnm_rs_mean)

# remove masked elements from cross-correlation matrix
#corr_mat_tvb_lsnm_rs = np.ma.compressed(corr_mat_tvb_lsnm_rs)

# plot a histogram to show the frequency of correlations
#plt.hist(corr_mat_tvb_lsnm_rs, 25)

#plt.xlabel('Correlation Coefficient')
#plt.ylabel('Number of occurrences')
#plt.axis([-1, 1, 0, 600])

#fig.savefig('tvb_lsnm_rs_hist_66_ROIs')

# calculate and print kurtosis
#print '\nTVB/LSNM Resting-State Fishers kurtosis: ', kurtosis(corr_mat_tvb_lsnm_rs, fisher=True)
#print 'TVB/LSNM Resting-State Skewness: ', skew(corr_mat_tvb_lsnm_rs)

# initialize new figure for tvb/lsnm resting-state histogram
#fig = plt.figure('TVB/LSNM Passive Viewing')
#cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
#mask = np.tri(tvb_lsnm_pv_mean.shape[0], k=0)
#mask = np.transpose(mask)
#tvb_lsnm_pv_mean = np.ma.array(tvb_lsnm_pv_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
#corr_mat_tvb_lsnm_pv = np.ma.ravel(tvb_lsnm_pv_mean)

# remove masked elements from cross-correlation matrix
#corr_mat_tvb_lsnm_pv = np.ma.compressed(corr_mat_tvb_lsnm_pv)

# plot a histogram to show the frequency of correlations
#plt.hist(corr_mat_tvb_lsnm_pv, 25)

#plt.xlabel('Correlation Coefficient')
#plt.ylabel('Number of occurrences')
#plt.axis([-1, 1, 0, 600])

#fig.savefig('tvb_lsnm_pv_hist_66_ROIs')

# calculate and print kurtosis
#print '\nTVB/LSNM Passive Viewing Fishers kurtosis: ', kurtosis(corr_mat_tvb_lsnm_pv, fisher=True)
#print 'TVB/LSNM Passive Viewing Skewness: ', skew(corr_mat_tvb_lsnm_pv)

# initialize new figure for tvb/lsnm resting-state histogram
#fig = plt.figure('TVB/LSNM DMS Task')
#cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
#mask = np.tri(tvb_lsnm_dms_mean.shape[0], k=0)
#mask = np.transpose(mask)
#tvb_lsnm_dms_mean = np.ma.array(tvb_lsnm_dms_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
#corr_mat_tvb_lsnm_dms = np.ma.ravel(tvb_lsnm_dms_mean)

# remove masked elements from cross-correlation matrix
#corr_mat_tvb_lsnm_dms = np.ma.compressed(corr_mat_tvb_lsnm_dms)

# plot a histogram to show the frequency of correlations
#plt.hist(corr_mat_tvb_lsnm_dms, 25)

#plt.xlabel('Correlation Coefficient')
#plt.ylabel('Number of occurrences')
#plt.axis([-1, 1, 0, 600])

#fig.savefig('tvb_lsnm_dms_hist_66_ROIs')

# calculate and print kurtosis
#print '\nTVB/LSNM DMS Task Fishers kurtosis: ', kurtosis(corr_mat_tvb_lsnm_dms, fisher=True)
#print 'TVB/LSNM DMS Task Skewness: ', skew(corr_mat_tvb_lsnm_dms)

# find the 10 highest correlation values in a cross-correlation matrix
#corr_mat_tvb_lsnm_dms_sorted = np.sort(corr_mat_tvb_lsnm_dms, axis=None)[::-1]
#np.set_printoptions(threshold='nan')
#print 'Sorted FCs in DMS mean array', corr_mat_tvb_lsnm_dms_sorted[:20]

# plot scatter plots to show correlations of TVB-RS vs TVB-LSNM-RS,
# TVB-LSNM RS vs TVB-LSNM PV, and TVB-LSNM PV vs TVB-LSNM DMS 

# start with a rectangular Figure
#fig=plt.figure('TVB-Only RS FC v TVB/LSNM RS FC')

# should we convert to Fisher's Z values prior to computing correlation of FCs?
#tvb_rs_z        = np.arctanh(corr_mat_tvb_rs)
#tvb_lsnm_rs_z   = np.arctanh(corr_mat_tvb_lsnm_rs)
#tvb_lsnm_pv_z   = np.arctanh(corr_mat_tvb_lsnm_pv)
#tvb_lsnm_dms_z  = np.arctanh(corr_mat_tvb_lsnm_dm)

# scatter plot
#plt.scatter(corr_mat_tvb_rs, corr_mat_tvb_lsnm_rs)
#plt.xlabel('TVB-Only RS FC')
#plt.ylabel('TVB/LSNM RS FC')
#plt.axis([-0.5,0.5,-0.5,1])

# fit scatter plot with np.polyfit
#m, b = np.polyfit(corr_mat_tvb_rs, corr_mat_tvb_lsnm_rs, 1)
#plt.plot(corr_mat_tvb_rs, m*corr_mat_tvb_rs + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(corr_mat_tvb_rs, corr_mat_tvb_lsnm_rs)[1,0]
#plt.text(-0.4, 0.75, 'r=' + '{:.2f}'.format(r))

#fig.savefig('rs_vs_rs_scatter_66_ROIs')

# start with a rectangular Figure
#fig=plt.figure('TVB/LSNM RS FC v TVB/LSNM PV FC')

# scatter plot
#plt.scatter(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_pv)
#plt.xlabel('TVB/LSNM RS FC')
#plt.ylabel('TVB/LSNM PV FC')
#plt.axis([-0.5,0.5,-0.5,1])

# fit scatter plot with np.polyfit
#m, b = np.polyfit(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_pv, 1)
#plt.plot(corr_mat_tvb_lsnm_rs, m*corr_mat_tvb_lsnm_rs + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_pv)[1,0]
#plt.text(-0.4, 0.75, 'r=' + '{:.2f}'.format(r))

#fig.savefig('rs_vs_pv_scatter_66_ROIs')

# start with a rectangular Figure
#fig=plt.figure('TVB/LSNM RS FC v TVB/LSNM DMS FC')

# scatter plot
#plt.scatter(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_dms)  
#plt.xlabel('TVB/LSNM RS FC')
#plt.ylabel('TVB/LSNM DMS FC')
#plt.axis([-0.5,0.5,-0.5,1])

# fit scatter plot with np.polyfit
#m, b = np.polyfit(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_dms, 1)
#plt.plot(corr_mat_tvb_lsnm_rs, m*corr_mat_tvb_lsnm_rs + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_dms)[1,0]
#plt.text(-0.4, 0.75, 'r=' + '{:.2f}'.format(r))

#fig.savefig('rs_vs_dms_scatter_66_ROIs')

############################################################################
# Plot correlations between degree of RS v PV and RS v DMS
###########################################################################
#fig=plt.figure('Degree correlation between TVB/LSNM RS PV DMS at various sparsities')

# compute correlation coefficients for all sparsities and conditions of interest
#r_array_RS_v_PV = np.zeros(num_of_densities)
#r_array_RS_v_DMS = np.zeros(num_of_densities)
#r_array_PV_v_DMS = np.zeros(num_of_densities)
#for d in range(0, num_of_densities):
#    r_array_RS_v_PV[d] = np.corrcoef(TVB_LSNM_RS_DEGREE[d], TVB_LSNM_PV_DEGREE[d])[1,0]
#    r_array_RS_v_DMS[d] = np.corrcoef(TVB_LSNM_RS_DEGREE[d], TVB_LSNM_DMS_DEGREE[d])[1,0]
#    r_array_PV_v_DMS[d] = np.corrcoef(TVB_LSNM_PV_DEGREE[d], TVB_LSNM_DMS_DEGREE[d])[1,0]

#print 'Each node degree array is of the following size: ', TVB_LSNM_RS_DEGREE[d].shape

#plt.plot(threshold_array, r_array_RS_v_PV,  label='TVB/LSNM RS v PV')
#plt.plot(threshold_array, r_array_RS_v_DMS, label='TVB/LSNM RS v DMS')
#plt.plot(threshold_array, r_array_PV_v_DMS, label='TVB/LSNM PV v DMS')
#plt.legend(loc='best')
#plt.xlabel('Sparsity')
#plt.ylabel('Correlation')

#fig.savefig('corr_RS_PV_DMS.png')

# initialize figure to plot degree correlations at sparsity 50%
#fig=plt.figure('Degree Correlation between TVB/LSNM RS and TVB/LSNM PV AT 50%')

#plt.scatter(TVB_LSNM_RS_DEGREE[-1], TVB_LSNM_PV_DEGREE[-1])
#plt.xlabel('TVB/LSNM RS DEGREE')
#plt.ylabel('TVB/LSNM PV DEGREE')

# fit scatter plot with np.polyfit
#m, b = np.polyfit(TVB_LSNM_RS_DEGREE[-1], TVB_LSNM_PV_DEGREE[-1], 1)
#plt.plot(TVB_LSNM_RS_DEGREE[-1], m*TVB_LSNM_RS_DEGREE[-1] + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(TVB_LSNM_RS_DEGREE[-1], TVB_LSNM_PV_DEGREE[-1])[1,0]
#plt.text(1, 22, 'r=' + '{:.2f}'.format(r))

# initialize figure to plot degree correlations at sparsity 10%
#fig=plt.figure('Degree Correlation between TVB/LSNM RS and TVB/LSNM DMS AT 50%')

#plt.scatter(TVB_LSNM_RS_DEGREE[-1], TVB_LSNM_DMS_DEGREE[-1])
#plt.xlabel('TVB/LSNM RS DEGREE')
#plt.ylabel('TVB/LSNM DMS DEGREE')

# fit scatter plot with np.polyfit
#m, b = np.polyfit(TVB_LSNM_RS_DEGREE[-1], TVB_LSNM_DMS_DEGREE[-1], 1)
#plt.plot(TVB_LSNM_RS_DEGREE[-1], m*TVB_LSNM_RS_DEGREE[-1] + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(TVB_LSNM_RS_DEGREE[-1], TVB_LSNM_DMS_DEGREE[-1])[1,0]
#plt.text(1, 22, 'r=' + '{:.2f}'.format(r))

#############################################################################
# Plot node degree difference as bar graphs
#############################################################################
# initialize new figure for node degree differences bar chart
#fig=plt.figure('Node degree differences between RS and DMS FC networks')
#bar_pos = np.arange(len(labels))
#plt.bar(bar_pos, DMS_RS_DEGREE_DIFF[1], align='center', alpha=0.5)
#plt.xticks(bar_pos, labels, rotation='vertical')
#plt.ylabel('Node degree difference')
#plt.xlabel('ROI')


###################################################################################################
# plot correlation coefficients of the simulated ROIs vs empirical ROIs
# first, find average of correlation coefficients within each lo-res ROI by averaging across hi-res
# ROIs located within each one of the 66 lo-res ROIs
##################################################################################################

# simulated RS FC needs to be rearranged prior to scatter plotting
#new_TVB_RS_FC = np.zeros([66, 66])
#for i in range(0, 66):
#    for j in range(0, 66):

        # extract labels of current ROI label from simulated labels list of simulated FC 
#        label_i = labels[i]
#        label_j = labels[j]
        # extract index of corresponding ROI label from empirical FC labels list
#        emp_i = np.where(hagmann_empirical['anat_lbls'] == label_i)[0][0]
#        emp_j = np.where(hagmann_empirical['anat_lbls'] == label_j)[0][0]
#        new_TVB_RS_FC[emp_i, emp_j] = tvb_rs_mean[i, j]

######################################################################
# Plot correlations between simulated TVB/LSNM RS and TVB/LSNM PV
# and between simulated TVB/LSNM RS abd TVB/LSNM DMS
######################################################################
# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_lsnm_lowres_rs_mean.shape[0], k=0)
mask = np.transpose(mask)
new_TVB_LSNM_RS_FC = np.ma.array(tvb_lsnm_lowres_rs_mean, mask=mask)    # mask out upper triangle
new_TVB_LSNM_PV_FC = np.ma.array(tvb_lsnm_lowres_pv_mean, mask=mask)    # mask out upper triangle
new_TVB_LSNM_DMS_FC= np.ma.array(tvb_lsnm_lowres_dms_mean,mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrices
corr_mat_TVB_LSNM_RS_FC  = np.ma.ravel(new_TVB_LSNM_RS_FC)
corr_mat_TVB_LSNM_PV_FC  = np.ma.ravel(new_TVB_LSNM_PV_FC)
corr_mat_TVB_LSNM_DMS_FC = np.ma.ravel(new_TVB_LSNM_DMS_FC)

# remove masked elements from cross-correlation matrix
corr_mat_TVB_LSNM_RS_FC = np.ma.compressed(corr_mat_TVB_LSNM_RS_FC)
corr_mat_TVB_LSNM_PV_FC = np.ma.compressed(corr_mat_TVB_LSNM_PV_FC)
corr_mat_TVB_LSNM_DMS_FC= np.ma.compressed(corr_mat_TVB_LSNM_DMS_FC)

# scatter plot: RS vs PV
fig=plt.figure('Correlation between TVB/LSNM RS and TVB/LSNM PV')
plt.scatter(corr_mat_TVB_LSNM_RS_FC, corr_mat_TVB_LSNM_PV_FC)
plt.xlabel('TVB/LSNM PF FC')
plt.ylabel('TVB/LSNM PV FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(corr_mat_TVB_LSNM_RS_FC, corr_mat_TVB_LSNM_PV_FC, 1)
plt.plot(corr_mat_TVB_LSNM_RS_FC, m*corr_mat_TVB_LSNM_RS_FC + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(corr_mat_TVB_LSNM_RS_FC, corr_mat_TVB_LSNM_PV_FC)[1,0]
plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

print 'Correlation between PF and PV is: ', r

fig.savefig('corr_rs_vs_pv.png')

# scatter plot: RS vs DMS
fig=plt.figure('Correlation between TVB/LSNM RS and TVB/LSNM DMS')
plt.scatter(corr_mat_TVB_LSNM_RS_FC, corr_mat_TVB_LSNM_DMS_FC)
plt.xlabel('TVB/LSNM PF FC')
plt.ylabel('TVB/LSNM DMS FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(corr_mat_TVB_LSNM_RS_FC, corr_mat_TVB_LSNM_DMS_FC, 1)
plt.plot(corr_mat_TVB_LSNM_RS_FC, m*corr_mat_TVB_LSNM_RS_FC + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(corr_mat_TVB_LSNM_RS_FC, corr_mat_TVB_LSNM_DMS_FC)[1,0]
plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

print 'Correlation between PF and DMS is: ', r

fig.savefig('corr_rs_vs_dms.png')


######################################################################
# Find and display the 10 highest correlations in each condition
#(PF, PV, DMS) using circular graphs
######################################################################
# First, dump contents of matrix into an array that includes origin, destination,
# and weight of each functional connection
PF_FC_list = np.zeros([lores_ROIs*lores_ROIs, 3])
PV_FC_list = np.zeros([lores_ROIs*lores_ROIs, 3])
DMS_FC_list= np.zeros([lores_ROIs*lores_ROIs, 3])
x=0
for i in range(0, tvb_lsnm_lowres_rs_mean.shape[0]):
    for j in range(0, tvb_lsnm_lowres_rs_mean.shape[1]):
        PF_FC_list[x, 0]  = i
        PF_FC_list[x, 1]  = j
        PF_FC_list[x, 2]  = tvb_lsnm_lowres_rs_mean[i, j]
        PV_FC_list[x, 0]  = i
        PV_FC_list[x, 1]  = j
        PV_FC_list[x, 2]  = tvb_lsnm_lowres_pv_mean[i, j]
        DMS_FC_list[x, 0] = i
        DMS_FC_list[x, 1] = j
        DMS_FC_list[x, 2] = tvb_lsnm_lowres_dms_mean[i, j]

        x = x + 1

# now we need to sort the lists in ascending order
PF_FC_list_sorted = np.argsort(PF_FC_list[:,2])
PV_FC_list_sorted = np.argsort(PV_FC_list[:,2])
DMS_FC_list_sorted= np.argsort(DMS_FC_list[:,2])

# store the 10 highest functional connections (start from last item)
# one look at every other item bc we have a symetrical matrix
print 'PF Top Ten FCs: '
for i in range(lores_ROIs*lores_ROIs-20,lores_ROIs*lores_ROIs, 2):
    idx = PF_FC_list_sorted[i]
    largest = PF_FC_list[idx, 2]
    source  = labels[int(PF_FC_list[idx, 0])]
    dest    = labels[int(PF_FC_list[idx, 1])]
    print '(', source, ',', dest, ',', format(largest, '.2f'), ')'

print 'PV Top Ten FCs: '
for i in range(lores_ROIs*lores_ROIs-20,lores_ROIs*lores_ROIs, 2):
    idx = PV_FC_list_sorted[i]
    largest = PV_FC_list[idx, 2]
    source  = labels[int(PV_FC_list[idx, 0])]
    dest    = labels[int(PV_FC_list[idx, 1])]
    print '(', source, ',', dest, ',', format(largest, '.2f'), ')'

print 'DMS Top Ten FCs: '
for i in range(lores_ROIs*lores_ROIs-20,lores_ROIs*lores_ROIs, 2):
    idx = DMS_FC_list_sorted[i]
    largest = DMS_FC_list[idx, 2]
    source  = labels[int(DMS_FC_list[idx, 0])]
    dest    = labels[int(DMS_FC_list[idx, 1])]
    print '(', source, ',', dest, ',', format(largest, '.2f'), ')'

label_names = labels
node_order = labels
node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

# We only show the strongest connections.
fig = plt.figure('PF FC strongest functional connections')
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111)
plot_connectivity_circle(tvb_lsnm_lowres_rs_mean, label_names, n_lines=10,
                         node_angles=node_angles, 
                         title='Strongest functional connections during PF',
                         vmin=-1, vmax=1, colormap='hot', fig=fig)
fig.savefig('circle_strongest_pf_fc.png', facecolor='black')

fig = plt.figure('PV FC strongest functional connections')
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111)
plot_connectivity_circle(tvb_lsnm_lowres_pv_mean, label_names, n_lines=10,
                         node_angles=node_angles, 
                         title='Strongest functional connections during PV',
                         vmin=-1, vmax=1, colormap='hot', fig=fig)
fig.savefig('circle_strongest_pv_fc.png', facecolor='black')

fig = plt.figure('TVB/LSNM PV FC')
fig.patch.set_facecolor('black')
ax = fig.add_subplot(111)
plot_connectivity_circle(tvb_lsnm_lowres_dms_mean, label_names, n_lines=10,
                         node_angles=node_angles, 
                         title='Strongest functional connections during DMS',
                         vmin=-1, vmax=1, colormap='hot', fig=fig)
fig.savefig('circle_strongest_dms_fc.png', facecolor='black')


###########################################################################
# plot PSC of average BOLD for all hires ROIs for each condition separately
###########################################################################
fig=plt.figure('High resolution fMRI BOLD timeseries - PF vs PV vs DMS')
t_secs = t_steps*2        # convert MR scans to seconds
avg_pf_bold = np.mean(tvb_lsnm_pf_bold, axis=0)
timecourse_mean_pf = np.mean(avg_pf_bold)
psc_pf = avg_pf_bold / timecourse_mean_pf * 100. - 100.
plt.plot(t_secs, psc_pf, label='PF')
plt.axvspan( 1, 15.5, facecolor='gray', alpha=0.2)
plt.axvspan(35, 49.5, facecolor='gray', alpha=0.2)
plt.axvspan(69, 83.5, facecolor='gray', alpha=0.2)
plt.axvspan(103, 117.5, facecolor='gray', alpha=0.2)
plt.axvspan(137, 151.5, facecolor='gray', alpha=0.2)
plt.axvspan(171, 185.5, facecolor='gray', alpha=0.2)

avg_pv_bold = np.mean(tvb_lsnm_pv_bold, axis=0)
timecourse_mean_pv = np.mean(avg_pv_bold)
psc_pv = avg_pv_bold / timecourse_mean_pv * 100. - 100.
plt.plot(t_secs, psc_pv, label='PV')

avg_dms_bold = np.mean(tvb_lsnm_dms_bold, axis=0)
timecourse_mean_dms = np.mean(avg_dms_bold)
psc_dms = avg_dms_bold / timecourse_mean_dms * 100. - 100.
plt.plot(t_secs, psc_dms, label='DMS')

plt.legend(loc='best')

###########################################################################
# plot z-scores of average BOLD for all low-res ROIs for each condition
###########################################################################
X = np.array([[1,1,1,1,1,1,1,      # design matrix
               0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,      
               0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,      
               0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,      
               0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,      
               0,0,0,0,0,0,0,0,0,
               1,1,1,1,1,1,1,
               0,0,0,0],
              [1,1,1,1,1,1,1,      
               1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,     
               1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,     
               1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,      
               1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,     
               1,1,1,1,1,1,1,1,1,
               1,1,1,1,1,1,1,
               1,1,1,1]])
X = X.T
print 'Shape of design matrix: ', X.shape
Y1 = tvb_lsnm_pv_bold_lowres.T          # observations matrix
Y2 = tvb_lsnm_dms_bold_lowres.T          # observations matrix
cval = np.array([1,0])              # contrast
print 'Shape of cval: ', cval.shape 
model1 = GeneralLinearModel(X)
model2 = GeneralLinearModel(X)
model1.fit(Y1)
model2.fit(Y2)
z_vals_1 = model1.contrast(cval).z_score() # z-transformed statistics
z_vals_2 = model2.contrast(cval).z_score() # z-transformed statistics

z_vals = np.vstack((z_vals_1, z_vals_2))

fig = plt.figure('z-values')
ax = fig.add_subplot(111)
# plot z-values as a heatmap
cax = ax.imshow(z_vals.T, interpolation='nearest', cmap='bwr', aspect='auto')
ax.grid(False)
color_bar=plt.colorbar(cax)
# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, 2))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(['PV', 'DMS'])
ax.set_yticklabels(labels)

######################################################################
# Plot correlations between empirical and simulated RS FC
#####################################################################
#fig=plt.figure('Empirical vs TVB RS FC')

# apply mask to get rid of upper triangle, including main diagonal
#mask = np.tri(tvb_rs_lowres_mean.shape[0], k=0)
#mask = np.transpose(mask)
#new_TVB_RS_FC = np.ma.array(tvb_rs_lowres_mean, mask=mask)           # mask out upper triangle
#empirical_fc_lowres = np.ma.array(empirical_fc_lowres, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
#corr_mat_sim_TVB_RS_FC = np.ma.ravel(new_TVB_RS_FC)
#corr_mat_emp_FC = np.ma.ravel(empirical_fc_lowres)

# remove masked elements from cross-correlation matrix
#corr_mat_sim_TVB_RS_FC = np.ma.compressed(corr_mat_sim_TVB_RS_FC)
#corr_mat_emp_FC = np.ma.compressed(corr_mat_emp_FC)

# scatter plot Empirical vs TVB RS
#plt.scatter(corr_mat_emp_FC, corr_mat_sim_TVB_RS_FC)
#plt.xlabel('Empirical FC')
#plt.ylabel('TVB RS FC')

# fit scatter plot with np.polyfit
#m, b = np.polyfit(corr_mat_emp_FC, corr_mat_sim_TVB_RS_FC, 1)
#plt.plot(corr_mat_emp_FC, m*corr_mat_emp_FC + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(corr_mat_emp_FC, corr_mat_sim_TVB_RS_FC)[1,0]
#plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

# scatter plot Empirical vs TVB/LSNM RS
#fig=plt.figure('Empirical vs TVB/LSNM RS FC')
#plt.scatter(corr_mat_emp_FC, corr_mat_TVB_LSNM_RS_FC)
#plt.xlabel('Empirical FC')
#plt.ylabel('TVB/LSNM RS FC')

# fit scatter plot with np.polyfit
#m, b = np.polyfit(corr_mat_emp_FC, corr_mat_TVB_LSNM_RS_FC, 1)
#plt.plot(corr_mat_emp_FC, m*corr_mat_emp_FC + b, '-', color='red')

# calculate correlation coefficient and display it on plot
#r = np.corrcoef(corr_mat_emp_FC, corr_mat_TVB_LSNM_RS_FC)[1,0]
#plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

###########################################################################
# Plot empirical structural connectivity matrices (lo-res and hi-res) that
# was used for simulations
###########################################################################
#fig=plt.figure('TVB/Hagmann Structural Connectivity Matrix (66 ROIs)')
#ax = fig.add_subplot(111)
#cmap = CM.get_cmap('jet', 10)
#cax = ax.imshow(empirical_sc_lowres, vmin=0, vmax=1, interpolation='nearest', cmap=cmap)
#ax.grid(False)
#color_bar=plt.colorbar(cax)
#fig.savefig('empirical_sc_lores.png')
#
#fig=plt.figure('TVB/Hagmann Structural Connectivity Matrix (998 ROIs)')
#ax = fig.add_subplot(111)
#cmap = CM.get_cmap('jet', 10)
#cax = ax.imshow(empirical_sc_hires, vmin=0, vmax=1, interpolation='nearest', cmap=cmap)
#ax.grid(False)
#color_bar=plt.colorbar(cax)
#fig.savefig('empirical_sc_hires.png')

# Show the plots on the screen
plt.show()
