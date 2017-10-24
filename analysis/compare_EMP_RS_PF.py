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
#   This file (compare_EMP_RS_PF.py) was created on October 18 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on October 18 2017
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
# compare_EMP_RS_PF.py
#
# Reads the correlation coefficients (functional connectivity matrix) from
# several python (*.npy) data files, each
# corresponding to a single subject, and calculates the average functional
# connectivity across all subjects for four conditions:
#     (a) Empirical Resting State,
#     (b) TVB-only Resting State,
#     (c) TVB-only Resting State with stimulation to visual nodes
#     (d) TVB/LSNM Passive Fixation

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
TVB_RS_w_stim_FC_avg_file = 'tvb_rs_w_stim_fc_avg.npy'
TVB_LSNM_PF_FC_avg_file = 'tvb_lsnm_pf_fc_avg.npy'
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
TVB_RS_subj1  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj2  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj3  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj4  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj5  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj6  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj7  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj8  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj9  = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_subj10 = 'subject_tvb/output.RestingState.198_3_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'

TVB_RS_w_stim_subj1  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj2  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj3  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj4  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj5  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj6  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj7  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj8  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj9  = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_subj10 = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/xcorr_matrix_998_regions_3T_0.25Hz.npy'

TVB_LSNM_PF_subj1  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj2  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj3  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj4  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj5  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj6  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj7  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj8  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj9  = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_subj10 = 'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'

TVB_LSNM_DMS_subj1  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj2  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj3  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj4  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj5  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj6  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj7  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj8  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj9  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_subj10 = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/xcorr_matrix_998_regions_3T_0.25Hz.npy'

# upload fMRI BOLD time series for all subject and conditions
TVB_RS_BOLD        = 'subject_tvb/output.RestingState.198_3_0.15/bold_balloon_998_regions_3T_0.25Hz.npy'
TVB_RS_w_stim_BOLD = 'subject_tvb/output.RestingState.198_3_0.15_w_stim/bold_balloon_998_regions_3T_0.25Hz.npy'
TVB_LSNM_PF_BOLD   = 'subject_12/output.Fixation_incl_PreSMA_3.0_0.15/bold_balloon_998_regions_3T_0.25Hz.npy'
TVB_LSNM_DMS_BOLD  = 'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/bold_balloon_998_regions_3T_0.25Hz.npy'

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

tvb_rs_w_stim_subj1  = np.load(TVB_RS_w_stim_subj1)
tvb_rs_w_stim_subj2  = np.load(TVB_RS_w_stim_subj2)
tvb_rs_w_stim_subj3  = np.load(TVB_RS_w_stim_subj3)
tvb_rs_w_stim_subj4  = np.load(TVB_RS_w_stim_subj4)
tvb_rs_w_stim_subj5  = np.load(TVB_RS_w_stim_subj5)
tvb_rs_w_stim_subj6  = np.load(TVB_RS_w_stim_subj6)
tvb_rs_w_stim_subj7  = np.load(TVB_RS_w_stim_subj7)
tvb_rs_w_stim_subj8  = np.load(TVB_RS_w_stim_subj8)
tvb_rs_w_stim_subj9  = np.load(TVB_RS_w_stim_subj9)
tvb_rs_w_stim_subj10 = np.load(TVB_RS_w_stim_subj10)

tvb_lsnm_pf_subj1  = np.load(TVB_LSNM_PF_subj1)
tvb_lsnm_pf_subj2  = np.load(TVB_LSNM_PF_subj2)
tvb_lsnm_pf_subj3  = np.load(TVB_LSNM_PF_subj3)
tvb_lsnm_pf_subj4  = np.load(TVB_LSNM_PF_subj4)
tvb_lsnm_pf_subj5  = np.load(TVB_LSNM_PF_subj5)
tvb_lsnm_pf_subj6  = np.load(TVB_LSNM_PF_subj6)
tvb_lsnm_pf_subj7  = np.load(TVB_LSNM_PF_subj7)
tvb_lsnm_pf_subj8  = np.load(TVB_LSNM_PF_subj8)
tvb_lsnm_pf_subj9  = np.load(TVB_LSNM_PF_subj9)
tvb_lsnm_pf_subj10 = np.load(TVB_LSNM_PF_subj10)

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
tvb_rs_bold  = np.load(TVB_RS_BOLD)
tvb_rs_w_stim_bold  = np.load(TVB_RS_w_stim_BOLD)
tvb_lsnm_pf_bold  = np.load(TVB_LSNM_PF_BOLD)
tvb_lsnm_dms_bold = np.load(TVB_LSNM_DMS_BOLD)

# construct numpy arrays that contain correlation coefficient arrays for all subjects
tvb_rs = np.array([tvb_rs_subj1, tvb_rs_subj2, tvb_rs_subj3,
                   tvb_rs_subj4, tvb_rs_subj5, tvb_rs_subj6,
                   tvb_rs_subj7, tvb_rs_subj8, tvb_rs_subj9,
                   tvb_rs_subj10 ]) 
tvb_rs_w_stim = np.array([tvb_rs_w_stim_subj1, tvb_rs_w_stim_subj2, tvb_rs_w_stim_subj3,
                          tvb_rs_w_stim_subj4, tvb_rs_w_stim_subj5, tvb_rs_w_stim_subj6,
                          tvb_rs_w_stim_subj7, tvb_rs_w_stim_subj8, tvb_rs_w_stim_subj9,
                          tvb_rs_w_stim_subj10 ]) 
tvb_lsnm_pf = np.array([tvb_lsnm_pf_subj1, tvb_lsnm_pf_subj2, tvb_lsnm_pf_subj3,
                        tvb_lsnm_pf_subj4, tvb_lsnm_pf_subj5, tvb_lsnm_pf_subj6,
                        tvb_lsnm_pf_subj7, tvb_lsnm_pf_subj8, tvb_lsnm_pf_subj9,
                        tvb_lsnm_pf_subj10 ]) 
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
tvb_rs_bold_lowres        = np.zeros([len(roi_dict), tvb_rs_bold.shape[1]])
tvb_rs_w_stim_bold_lowres = np.zeros([len(roi_dict), tvb_rs_w_stim_bold.shape[1]])
tvb_lsnm_pf_bold_lowres   = np.zeros([len(roi_dict), tvb_lsnm_pf_bold.shape[1]])
tvb_lsnm_dms_bold_lowres  = np.zeros([len(roi_dict), tvb_lsnm_dms_bold.shape[1]])

idx=0
for roi in roi_dict:
    for t in t_steps:
        tvb_rs_bold_lowres[idx, t] = np.mean(tvb_rs_bold[roi_dict[roi], t])
        tvb_rs_w_stim_bold_lowres[idx, t] = np.mean(tvb_rs_w_stim_bold[roi_dict[roi], t])
        tvb_lsnm_pf_bold_lowres[idx, t]  = np.mean(tvb_lsnm_pf_bold[roi_dict[roi], t])
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
tvb_rs_w_stim_z   = np.arctanh(tvb_rs_w_stim[:,0:hires_ROIs, 0:hires_ROIs])
tvb_lsnm_pf_z   = np.arctanh(tvb_lsnm_pf[:,0:hires_ROIs, 0:hires_ROIs])
tvb_lsnm_dms_z  = np.arctanh(tvb_lsnm_dms[:,0:hires_ROIs, 0:hires_ROIs])

# initialize 66x66 matrices
tvb_rs_lowres_Z = np.zeros([num_of_sub, lores_ROIs, lores_ROIs])
tvb_rs_w_stim_lowres_Z = np.zeros([num_of_sub, lores_ROIs, lores_ROIs])
tvb_lsnm_pf_lowres_Z = np.zeros([num_of_sub, lores_ROIs, lores_ROIs])
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
            tvb_rs_w_stim_lowres_Z[s, x, y]  += tvb_rs_w_stim_z[s, i, j]
            tvb_lsnm_pf_lowres_Z[s, x, y]  += tvb_lsnm_pf_z[s, i, j]
            tvb_lsnm_dms_lowres_Z[s, x, y] += tvb_lsnm_dms_z[s, i, j]

    # average the sum of each bucket dividing by no. of items in each bucket
    for x in range(0, lores_ROIs):
        for y in range(0, lores_ROIs):
            total_freq = freq_array[x][1] * freq_array[y][1]
            tvb_rs_lowres_Z[s, x, y] =      tvb_rs_lowres_Z[s, x, y] / total_freq
            tvb_rs_w_stim_lowres_Z[s, x, y] = tvb_rs_w_stim_lowres_Z[s, x, y] / total_freq 
            tvb_lsnm_pf_lowres_Z[s, x, y] = tvb_lsnm_pf_lowres_Z[s, x, y] / total_freq
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
# calculate binary SC matrix for a lo-res 66 ROI set (derived from 998X998 sc)
#############################################################################
bin_66_ROI_sc = np.zeros([lores_ROIs, lores_ROIs])
for i in range(0, hires_ROIs):
    for j in range(0, hires_ROIs):

        # extract low-res coordinates from hi-res empirical labels matrix
        x = hagmann_empirical['roi_lbls'][0][i]
        y = hagmann_empirical['roi_lbls'][0][j]

        if white_matter.weights[i, j] > 0.:
            bin_66_ROI_sc[x, y] = 1


#############################################################################
# Average simuated FC matrices across subjects
#############################################################################
print 'Averaging FC matrices across subjects...'

# calculate the mean of correlation coefficients across all given subjects
# for the high resolution (998x998) correlation matrices
tvb_rs_z_mean = np.mean(tvb_rs_z, axis=0)
tvb_rs_w_stim_z_mean = np.mean(tvb_rs_w_stim_z, axis=0)
tvb_lsnm_pf_z_mean = np.mean(tvb_lsnm_pf_z, axis=0)
tvb_lsnm_dms_z_mean = np.mean(tvb_lsnm_dms_z, axis=0)

# calculate the mean of correlation coefficients across all given subjects
# for the low resolution (66x66) correlation matrices
tvb_rs_lowres_Z_mean = np.mean(tvb_rs_lowres_Z, axis=0)
tvb_rs_w_stim_lowres_Z_mean = np.mean(tvb_rs_w_stim_lowres_Z, axis=0)
tvb_lsnm_pf_lowres_Z_mean = np.mean(tvb_lsnm_pf_lowres_Z, axis=0)
tvb_lsnm_dms_lowres_Z_mean = np.mean(tvb_lsnm_dms_lowres_Z, axis=0)

############################################################################
# Calculate differences between RS and PV and RS and DMS mean FC matrices
############################################################################
print 'Calculating differences between RS and Task matrices...'

rs_w_stim_minus_rs_Z =  tvb_rs_w_stim_lowres_Z_mean  - tvb_rs_lowres_Z_mean
pf_minus_rs_w_stim_Z = tvb_lsnm_pf_lowres_Z_mean - tvb_rs_w_stim_lowres_Z_mean
dms_minus_pf_Z = tvb_lsnm_dms_lowres_Z_mean - tvb_lsnm_pf_lowres_Z_mean

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
tvb_lowres_rs_mean  = np.tanh(tvb_rs_lowres_Z_mean)
tvb_lowres_rs_w_stim_mean  = np.tanh(tvb_rs_w_stim_lowres_Z_mean)
tvb_lsnm_lowres_pf_mean  = np.tanh(tvb_lsnm_pf_lowres_Z_mean)
tvb_lsnm_lowres_dms_mean  = np.tanh(tvb_lsnm_dms_lowres_Z_mean)

# now, convert back to from Z to R correlation coefficients (hi-res)
tvb_rs_mean  = np.tanh(tvb_rs_z_mean)
tvb_rs_w_stim_mean  = np.tanh(tvb_rs_w_stim_z_mean)
tvb_lsnm_pf_mean  = np.tanh(tvb_lsnm_pf_z_mean)
tvb_lsnm_dms_mean  = np.tanh(tvb_lsnm_dms_z_mean)

# also convert matrices of differences to R correlation coefficients 
rs_w_stim_minus_rs  = np.tanh(rs_w_stim_minus_rs_Z)
pf_minus_rs_w_stim = np.tanh(pf_minus_rs_w_stim_Z)
dms_minus_pf = np.tanh(dms_minus_pf_Z)

# fill diagonals of all matrices with zeros
np.fill_diagonal(empirical_fc_lowres, 0)
np.fill_diagonal(tvb_lowres_rs_mean, 0)
np.fill_diagonal(tvb_lowres_rs_w_stim_mean, 0)
np.fill_diagonal(tvb_lsnm_lowres_pf_mean, 0)
np.fill_diagonal(tvb_lsnm_lowres_dms_mean, 0)
np.fill_diagonal(tvb_rs_mean, 0)
np.fill_diagonal(tvb_rs_w_stim_mean, 0)
np.fill_diagonal(tvb_lsnm_pf_mean, 0)
np.fill_diagonal(tvb_lsnm_dms_mean, 0)
np.fill_diagonal(rs_w_stim_minus_rs, 0)
np.fill_diagonal(pf_minus_rs_w_stim, 0)
np.fill_diagonal(dms_minus_pf, 0)

# calculate the max and min of FC matrices and store in variables for later use
FC_max = max([np.amax(tvb_lowres_rs_mean),
              np.amax(tvb_lowres_rs_w_stim_mean),
              np.amax(tvb_lsnm_lowres_pf_mean),
              np.amax(tvb_lsnm_lowres_dms_mean)])
FC_diff_max = max([np.amax(rs_w_stim_minus_rs),
                   np.amax(pf_minus_rs_w_stim),
                   np.amax(dms_minus_pf)])
FC_emp_max = np.amax(empirical_fc_lowres)
FC_min = min([np.amin(tvb_lowres_rs_mean),
              np.amin(tvb_lowres_rs_w_stim_mean),
              np.amin(tvb_lsnm_lowres_pf_mean),
              np.amin(tvb_lsnm_lowres_dms_mean)])
FC_diff_min = max([np.amin(rs_w_stim_minus_rs),
                   np.amin(pf_minus_rs_w_stim),
                   np.amin(dms_minus_pf)])
FC_emp_min = np.amin(empirical_fc_lowres)
FC_max_diff_rs_w_stim_rs = np.sum(np.absolute(rs_w_stim_minus_rs), axis=1)
FC_max_diff_pf_rs_w_stim  = np.sum(np.absolute(pf_minus_rs_w_stim), axis=1)
FC_max_diff_dms_pf = np.sum(np.absolute(dms_minus_pf), axis=1)
FC_max_diff_idx_pf_rs_w_stim  = np.where(FC_max_diff_pf_rs_w_stim  == np.amax(FC_max_diff_pf_rs_w_stim))
FC_max_diff_idx_dms_pf = np.where(FC_max_diff_dms_pf == np.amax(FC_max_diff_dms_pf))

# save FC matrices to outputfiles
print 'Saving FC matrices for later use...'
np.save(EMP_RS_FC_file, empirical_fc_lowres)
np.save(TVB_RS_FC_avg_file, tvb_lowres_rs_mean)
np.save(TVB_RS_w_stim_FC_avg_file, tvb_lowres_rs_w_stim_mean)
np.save(TVB_LSNM_PF_FC_avg_file, tvb_lsnm_lowres_pf_mean)
np.save(TVB_LSNM_DMS_FC_avg_file, tvb_lsnm_lowres_dms_mean)

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

TVB_RS_w_stim_EFFICIENCY_G = np.zeros(num_sparsity)
TVB_RS_w_stim_EFFICIENCY_L = np.zeros(num_sparsity)
TVB_RS_w_stim_CLUSTERING = np.zeros(num_sparsity)
TVB_RS_w_stim_CHARPATH   = np.zeros(num_sparsity)
TVB_RS_w_stim_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
TVB_RS_w_stim_BTW_CENTRALITY = np.zeros(num_sparsity)
TVB_RS_w_stim_PARTICIPATION = np.zeros(num_sparsity)
TVB_RS_w_stim_SMALL_WORLDNESS = np.zeros(num_sparsity)
TVB_RS_w_stim_MODULARITY = np.zeros(num_sparsity)
#TVB_LSNM_RS_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
TVB_RS_w_stim_BW_RATIO  = np.zeros(num_sparsity)

TVB_LSNM_PF_EFFICIENCY_G = np.zeros(num_sparsity)
TVB_LSNM_PF_EFFICIENCY_L = np.zeros(num_sparsity)
TVB_LSNM_PF_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_PF_CHARPATH   = np.zeros(num_sparsity)
TVB_LSNM_PF_EIGEN_CENTRALITY   = np.zeros(num_sparsity)
TVB_LSNM_PF_BTW_CENTRALITY = np.zeros(num_sparsity)
TVB_LSNM_PF_PARTICIPATION = np.zeros(num_sparsity)
TVB_LSNM_PF_SMALL_WORLDNESS = np.zeros(num_sparsity)
TVB_LSNM_PF_MODULARITY = np.zeros(num_sparsity)
#TVB_LSNM_PF_DEGREE     = np.zeros((num_sparsity, lores_ROIs))
TVB_LSNM_PF_BW_RATIO  = np.zeros(num_sparsity)

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
tvb_rs_w_stim_bin = np.zeros((num_of_densities, lores_ROIs, lores_ROIs))
tvb_lsnm_pf_bin = np.zeros((num_of_densities, lores_ROIs, lores_ROIs))
tvb_lsnm_dms_bin= np.zeros((num_of_densities, lores_ROIs, lores_ROIs))

emp_rs_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_rs_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_rs_w_stim_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
tvb_lsnm_pf_hr_bin = np.zeros((num_of_densities, hires_ROIs, hires_ROIs))
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

    tvb_rs_a           = np.absolute(tvb_lowres_rs_mean)
    tvb_rs_p           = bct.threshold_proportional(tvb_rs_a, threshold_array[d], copy=True)
    tvb_rs_bin[d]      = bct.binarize(tvb_rs_p, copy=True)

    tvb_rs_a           = np.absolute(tvb_rs_mean)
    tvb_rs_p           = bct.threshold_proportional(tvb_rs_a, threshold_array[d], copy=True)
    tvb_rs_hr_bin[d]   = bct.binarize(tvb_rs_p, copy=True)
    
    tvb_rs_w_stim_a      = np.absolute(tvb_lowres_rs_w_stim_mean)
    tvb_rs_w_stim_p      = bct.threshold_proportional(tvb_rs_w_stim_a, threshold_array[d], copy=True)
    tvb_rs_w_stim_bin[d] = bct.binarize(tvb_rs_w_stim_p, copy=True)

    tvb_rs_w_stim_a         = np.absolute(tvb_rs_w_stim_mean)
    tvb_rs_w_stim_p         = bct.threshold_proportional(tvb_rs_w_stim_a, threshold_array[d], copy=True)
    tvb_rs_w_stim_hr_bin[d] = bct.binarize(tvb_rs_w_stim_p, copy=True)

    tvb_lsnm_pf_a      = np.absolute(tvb_lsnm_lowres_pf_mean)
    tvb_lsnm_pf_p      = bct.threshold_proportional(tvb_lsnm_pf_a, threshold_array[d], copy=True)
    tvb_lsnm_pf_bin[d] = bct.binarize(tvb_lsnm_pf_p, copy=True)

    tvb_lsnm_pf_a      = np.absolute(tvb_lsnm_pf_mean)
    tvb_lsnm_pf_p      = bct.threshold_proportional(tvb_lsnm_pf_a, threshold_array[d], copy=True)
    tvb_lsnm_pf_hr_bin[d]= bct.binarize(tvb_lsnm_pf_p, copy=True)
    
    tvb_lsnm_dms_a      = np.absolute(tvb_lsnm_lowres_dms_mean)
    tvb_lsnm_dms_p      = bct.threshold_proportional(tvb_lsnm_dms_a, threshold_array[d], copy=True)
    tvb_lsnm_dms_bin[d] = bct.binarize(tvb_lsnm_dms_p, copy=True)

    tvb_lsnm_dms_a      = np.absolute(tvb_lsnm_dms_mean)
    tvb_lsnm_dms_p      = bct.threshold_proportional(tvb_lsnm_dms_a, threshold_array[d], copy=True)
    tvb_lsnm_dms_hr_bin[d] = bct.binarize(tvb_lsnm_dms_p, copy=True)

    # calculate global efficiency for each condition using Brain Connectivity Toolbox
    EMP_RS_EFFICIENCY_G[d]        = bct.efficiency_bin(emp_rs_bin[d])  
    TVB_RS_EFFICIENCY_G[d]        = bct.efficiency_bin(tvb_rs_bin[d]) 
    TVB_RS_w_stim_EFFICIENCY_G[d] = bct.efficiency_bin(tvb_rs_w_stim_bin[d]) 
    TVB_LSNM_PF_EFFICIENCY_G[d]   = bct.efficiency_bin(tvb_lsnm_pf_bin[d])
    TVB_LSNM_DMS_EFFICIENCY_G[d]  = bct.efficiency_bin(tvb_lsnm_dms_bin[d]) 

    # calculate mean clustering coefficient using Brain Connectivity Toolbox
    EMP_RS_CLUSTERING[d]        = np.mean(bct.clustering_coef_bu(emp_rs_bin[d]))
    TVB_RS_CLUSTERING[d]        = np.mean(bct.clustering_coef_bu(tvb_rs_bin[d]))
    TVB_RS_w_stim_CLUSTERING[d] = np.mean(bct.clustering_coef_bu(tvb_rs_w_stim_bin[d]))
    TVB_LSNM_PF_CLUSTERING[d]   = np.mean(bct.clustering_coef_bu(tvb_lsnm_pf_bin[d]))
    TVB_LSNM_DMS_CLUSTERING[d]  = np.mean(bct.clustering_coef_bu(tvb_lsnm_dms_bin[d]))

    # calculate modularity using Brain Connectivity Toolbox
    emp_modularity             = bct.modularity_und(emp_rs_bin[d], gamma=1, kci=None)
    tvb_rs_modularity          = bct.modularity_und(tvb_rs_bin[d], gamma=1, kci=None)
    tvb_rs_w_stim_modularity     = bct.modularity_und(tvb_rs_w_stim_bin[d], gamma=1, kci=None)
    tvb_lsnm_pf_modularity     = bct.modularity_und(tvb_lsnm_pf_bin[d], gamma=1, kci=None)
    tvb_lsnm_dms_modularity    = bct.modularity_und(tvb_lsnm_dms_bin[d], gamma=1, kci=None)
    EMP_RS_MODULARITY[d]       = emp_modularity[1]
    TVB_RS_MODULARITY[d]       = tvb_rs_modularity[1]
    TVB_RS_w_stim_MODULARITY[d]  = tvb_rs_w_stim_modularity[1]
    TVB_LSNM_PF_MODULARITY[d]  = tvb_lsnm_pf_modularity[1]
    TVB_LSNM_DMS_MODULARITY[d] = tvb_lsnm_dms_modularity[1]


###############################################################################
# Display heatmaps of all low-res FC matrices, weighted and binarized,
# empirical and simulated for all conditions
###############################################################################
fig = plt.figure('Empirical RS FC matrix (binarized at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(emp_rs_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)
# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('binary_empirical_fc.png')

fig = plt.figure('Mean TVB-only RS FC')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(tvb_lowres_rs_mean,
                #vmin=-0.5, vmax=0.5,
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

fig.savefig('mean_tvb_only_rs_fc.png')

fig = plt.figure('Mean TVB RS With Stimulation FC')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lowres_rs_w_stim_mean,
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

fig.savefig('mean_tvb_rs_w_stim_fc_incl_PreSMA.png')

fig = plt.figure('Mean TVB/LSNM PF FC')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_lowres_pf_mean,
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

fig.savefig('mean_tvb_lsnm_pf_fc_incl_PreSMA.png')

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

fig.savefig('mean_tvb_lsnm_dms_fc_incl_PreSMA.png')

fig=plt.figure('Functional Connectivity Matrix of empirical BOLD (66 ROIs)')
ax = fig.add_subplot(111)
empirical_fc_hires = np.asarray(empirical_fc_hires)
cax = ax.imshow(empirical_fc_lowres,
                #vmin=-0.5, vmax=0.5,
                interpolation='nearest', cmap='bwr')
ax.grid(False)
color_bar=plt.colorbar(cax)
fig.savefig('empirical_rs_fc.png')

fig = plt.figure('Mean TVB-only RS FC (binary at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_rs_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)
fig.savefig('binary_tvb_only_rs_fc.png')

fig = plt.figure('Mean TVB/LSNM RS FC with Stimulation (binarized at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_rs_w_stim_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('binary_tvb_rs_w_stim_fc_incl_PreSMA.png')

fig = plt.figure('Mean TVB/LSNM PF FC (binarized at 40% sparsity)')
ax = fig.add_subplot(111)
# plot correlation matrix as a heatmap
cax = ax.imshow(tvb_lsnm_pf_bin[-1], interpolation='nearest', cmap='Greys')
ax.grid(False)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# decrease font size
plt.rcParams.update({'font.size': 9})

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

fig.savefig('binary_tvb_lsnm_pf_fc_incl_PreSMA.png')

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

fig.savefig('binary_tvb_lsnm_dms_fc_incl_PreSMA.png')

####################################################################################
# Display graph theory metrics of Emp, RS, PF, for all sparsity thresholds
####################################################################################
fig = plt.figure('Global Efficiency')
cax = fig.add_subplot(111)
plt.plot(threshold_array, EMP_RS_EFFICIENCY_G, label='Emp RS')
plt.plot(threshold_array, TVB_RS_EFFICIENCY_G, label='TVB RS')
plt.plot(threshold_array, TVB_RS_w_stim_EFFICIENCY_G, label='TVB RS with Stim')
plt.plot(threshold_array, TVB_LSNM_PF_EFFICIENCY_G, label='TVB/LSNM PF')
#plt.plot(threshold_array, RAND_MAT_EFFICIENCY, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Mean Global Efficiency')
plt.legend(loc='best')
fig.savefig('rs_pf_dms_global_efficiency_across_thresholds.png')

fig = plt.figure('Mean Clustering Coefficient')
cax = fig.add_subplot(111)
plt.plot(threshold_array, EMP_RS_CLUSTERING, label='Emp RS')
plt.plot(threshold_array, TVB_RS_CLUSTERING, label='TVB RS')
plt.plot(threshold_array, TVB_RS_w_stim_CLUSTERING, label='TVB RS with Stim')
plt.plot(threshold_array, TVB_LSNM_PF_CLUSTERING, label='TVB/LSNM PF')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Mean Clustering')
plt.legend(loc='best')
fig.savefig('rs_pf_dms_clustering_across_thresholds.png')

fig = plt.figure('Modularity')
cax = fig.add_subplot(111)
plt.plot(threshold_array, EMP_RS_MODULARITY, label='Emp RS')
plt.plot(threshold_array, TVB_RS_MODULARITY, label='TVB RS')
plt.plot(threshold_array, TVB_RS_w_stim_MODULARITY, label='TVB RS with Stim')
plt.plot(threshold_array, TVB_LSNM_PF_MODULARITY, label='TVB/LSNM PF')
#plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
plt.xlabel('Density threshold')
plt.ylabel('Modularity')
plt.legend(loc='best')
fig.savefig('rs_pf_dms_modularity_across_thresholds.png')

######################################################################
# Plot correlations between simulated TVB/LSNM RS and TVB/LSNM PV
# and between simulated TVB/LSNM RS abd TVB/LSNM DMS
######################################################################
# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_lowres_rs_mean.shape[0], k=0)
mask = np.transpose(mask)

empirical_fc_lowres_m = np.ma.array(empirical_fc_lowres, mask=mask)    # mask out upper triangle
bin_66_ROI_sc_m = np.ma.array(bin_66_ROI_sc, mask=mask)     # mask out upper triangle
new_TVB_RS_FC = np.ma.array(tvb_lowres_rs_mean, mask=mask)    # mask out upper triangle
new_TVB_RS_w_stim_FC = np.ma.array(tvb_lowres_rs_w_stim_mean, mask=mask)    # mask out upper triangle
new_TVB_LSNM_PF_FC = np.ma.array(tvb_lsnm_lowres_pf_mean, mask=mask)    # mask out upper triangle
new_TVB_LSNM_DMS_FC= np.ma.array(tvb_lsnm_lowres_dms_mean,mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrices
corr_mat_emp_FC = np.ma.ravel(empirical_fc_lowres_m)
flat_bin_66_ROI_sc = np.ma.ravel(bin_66_ROI_sc_m)
corr_mat_TVB_RS_FC  = np.ma.ravel(new_TVB_RS_FC)
corr_mat_TVB_RS_w_stim_FC  = np.ma.ravel(new_TVB_RS_w_stim_FC)
corr_mat_TVB_LSNM_PF_FC  = np.ma.ravel(new_TVB_LSNM_PF_FC)
corr_mat_TVB_LSNM_DMS_FC = np.ma.ravel(new_TVB_LSNM_DMS_FC)

# remove masked elements from cross-correlation matrix
corr_mat_emp_FC = np.ma.compressed(corr_mat_emp_FC)
flat_bin_66_ROI_sc = np.ma.compressed(flat_bin_66_ROI_sc)
corr_mat_TVB_RS_FC = np.ma.compressed(corr_mat_TVB_RS_FC)
corr_mat_TVB_RS_w_stim_FC = np.ma.compressed(corr_mat_TVB_RS_w_stim_FC)
corr_mat_TVB_LSNM_PF_FC = np.ma.compressed(corr_mat_TVB_LSNM_PF_FC)
corr_mat_TVB_LSNM_DMS_FC= np.ma.compressed(corr_mat_TVB_LSNM_DMS_FC)

# now, construct a mask using the binarized 66 ROI SC array.
# THis mask will be used to calculate correlations between FC connectivity
# matrices only when a structural connection exists between two regions
sc_mask = flat_bin_66_ROI_sc == 0                              # create mask
corr_mat_emp_FC_m = np.ma.array(corr_mat_emp_FC, mask=sc_mask) # apply mask
corr_mat_TVB_RS_FC_m = np.ma.array(corr_mat_TVB_RS_FC, mask=sc_mask) # apply mask
corr_mat_TVB_RS_w_stim_FC_m = np.ma.array(corr_mat_TVB_RS_w_stim_FC, mask=sc_mask) # apply mask
corr_mat_TVB_LSNM_PF_FC_m = np.ma.array(corr_mat_TVB_LSNM_PF_FC, mask=sc_mask) # apply mask
corr_mat_TVB_LSNM_DMS_FC_m = np.ma.array(corr_mat_TVB_LSNM_DMS_FC, mask=sc_mask) # apply mask
flat_emp_fc = np.ma.compressed(corr_mat_emp_FC_m)              # remove masked elements
flat_TVB_RS_fc = np.ma.compressed(corr_mat_TVB_RS_FC_m)              # remove masked elements
flat_TVB_RS_w_stim_fc = np.ma.compressed(corr_mat_TVB_RS_w_stim_FC_m)   # remove masked elements
flat_TVB_LSNM_PF_fc = np.ma.compressed(corr_mat_TVB_LSNM_PF_FC_m)    # remove masked elements
flat_TVB_LSNM_DMS_fc = np.ma.compressed(corr_mat_TVB_LSNM_DMS_FC_m)  # remove masked elements

# scatter plot: Empirical RS vs TVB RS 
fig=plt.figure('Correlation between Empirical RS and TVB RS')
plt.scatter(flat_emp_fc, flat_TVB_RS_fc)
plt.xlabel('TVB Emp RS FC')
plt.ylabel('TVB RS FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_emp_fc, flat_TVB_RS_fc, 1)
plt.plot(flat_emp_fc, m*flat_emp_fc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(flat_emp_fc, flat_TVB_RS_fc)[1,0]
plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

print 'Correlation between Empirical RS and TVB RS: ', r

fig.savefig('corr_emp_rs_vs_rs.png')

# scatter plot: RS vs RS w/ stim
fig=plt.figure('Correlation between TVB RS and TVB RS w stim')
plt.scatter(flat_TVB_RS_fc, flat_TVB_RS_w_stim_fc)
plt.xlabel('TVB RS FC')
plt.ylabel('TVB RS w stim FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_TVB_RS_fc, flat_TVB_RS_w_stim_fc, 1)
plt.plot(flat_TVB_RS_fc, m*flat_TVB_RS_fc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(flat_TVB_RS_fc, flat_TVB_RS_w_stim_fc)[1,0]
plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

print 'Correlation between TVB RS and TVB RS w Stim is: ', r

fig.savefig('corr_rs_vs_rs_w_stim.png')

# scatter plot: RS w stim vs PF
fig=plt.figure('Correlation between TVB RS w Stimulation and TVB/LSNM PF')
plt.scatter(flat_TVB_RS_w_stim_fc, flat_TVB_LSNM_PF_fc)
plt.xlabel('TVB RS w Stim FC')
plt.ylabel('TVB/LSNM PF FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_TVB_RS_w_stim_fc, flat_TVB_LSNM_PF_fc, 1)
plt.plot(flat_TVB_RS_w_stim_fc, m*flat_TVB_RS_w_stim_fc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(flat_TVB_RS_w_stim_fc, flat_TVB_LSNM_PF_fc)[1,0]
plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

print 'Correlation between TVB RS w Stim and TVB/LSNM PF is: ', r

fig.savefig('corr_rs_w_stim_vs_pf.png')

# scatter plot: PF vs DMS
fig=plt.figure('Correlation between TVB/LSNM PF and TVB/LSNM DMS')
plt.scatter(flat_TVB_LSNM_PF_fc, flat_TVB_LSNM_DMS_fc)
plt.xlabel('TVB/LSNM PF FC')
plt.ylabel('TVB/LSNM DMS FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_TVB_LSNM_PF_fc, flat_TVB_LSNM_DMS_fc, 1)
plt.plot(flat_TVB_LSNM_PF_fc, m*flat_TVB_LSNM_PF_fc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(flat_TVB_LSNM_PF_fc, flat_TVB_LSNM_DMS_fc)[1,0]
plt.text(0.5, -0.04, 'r=' + '{:.2f}'.format(r))

print 'Correlation between TVB/LSNM PF and TVB/LSNM DMS is: ', r

fig.savefig('corr_pf_vs_dms.png')


###########################################################################
# plot PSC of average BOLD for all hires ROIs for each condition separately
###########################################################################
fig=plt.figure('High resolution fMRI BOLD timeseries - RS vs RS w stim vs PF vs DMS')
t_secs = t_steps*2        # convert MR scans to seconds

avg_rs_bold = np.mean(tvb_rs_bold, axis=0)
timecourse_mean_rs = np.mean(avg_rs_bold)
psc_rs = avg_rs_bold / timecourse_mean_rs * 100. - 100.
plt.plot(t_secs, psc_rs, label='RS')

avg_rs_w_stim_bold = np.mean(tvb_rs_w_stim_bold, axis=0)
timecourse_mean_rs_w_stim = np.mean(avg_rs_w_stim_bold)
psc_rs_w_stim = avg_rs_w_stim_bold / timecourse_mean_rs_w_stim * 100. - 100.
plt.plot(t_secs, psc_rs_w_stim, label='RS w stim')

plt.axvspan( 1, 15.5, facecolor='gray', alpha=0.2)
plt.axvspan(35, 49.5, facecolor='gray', alpha=0.2)
plt.axvspan(69, 83.5, facecolor='gray', alpha=0.2)
plt.axvspan(103, 117.5, facecolor='gray', alpha=0.2)
plt.axvspan(137, 151.5, facecolor='gray', alpha=0.2)
plt.axvspan(171, 185.5, facecolor='gray', alpha=0.2)

avg_pf_bold = np.mean(tvb_lsnm_pf_bold, axis=0)
timecourse_mean_pf = np.mean(avg_pf_bold)
psc_pf = avg_pf_bold / timecourse_mean_pf * 100. - 100.
plt.plot(t_secs, psc_pf, label='PF')

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
Y1 = tvb_lsnm_pf_bold_lowres.T          # observations matrix
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
ax.set_xticklabels(['PF', 'DMS'])
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
