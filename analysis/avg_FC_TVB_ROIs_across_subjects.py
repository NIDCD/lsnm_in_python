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
#   This file (avg_FC_TVB_ROIs_across_subjects.py) was created on November 30 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on January 18 2016
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
# avg_FC_TVB_ROIs_across_subjects.py
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

import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.tools as tls

import matplotlib as mpl

import pandas as pd

from scipy.stats import t

from scipy import stats

import math as m

from scipy.stats import kurtosis

from scipy.stats import skew

from matplotlib import cm as CM

from mne.viz import circular_layout, plot_connectivity_circle

import bct as bct

import csv

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

# define output file where means, standard deviations, and variances will be stored
RS_FC_avg_file = 'tvb_only_rs_fc_avg.npy'
TB_FC_avg_file = 'tb_fc_avg.npy'

# declare ROI labels
labels =  ['rLOF',     
    'rPORB',         
    'rFP'  ,          
    'rMOF' ,          
    'rPTRI',          
    'rPOPE',          
    'rRMF' ,          
    'rSF'  ,          
    'rCMF' ,          
    'rPREC',          
    'rPARC',          
    'rRAC' ,          
    'rCAC' ,          
    'rPC'  ,          
    'rISTC',          
    'rPSTC',          
    'rSMAR',          
    'rSP'  ,          
    'rIP'  ,          
    'rPCUN',          
    'rCUN' ,          
    'rPCAL',          
    'rLOCC',          
    'rLING',          
    'rFUS' ,          
    'rPARH',          
    'rENT' ,          
    'rTP'  ,          
    'rIT'  ,          
    'rMT'  ,          
    'rBSTS',          
    'rST'  ,          
    'rTT',
    'lLOF' ,
    'lPORB',
    'lFP'  ,
    'lMOF' ,
    'lPTRI',
    'lPOPE',
    'LRMF' ,
    'lSF'  ,
    'lCMF' ,
    'lPREC',
    'lPARC',
    'lRAC' ,
    'lCAC' ,
    'lPC'  ,
    'lISTC',
    'lPSTC',
    'lSMAR',
    'lSP'  ,
    'lIP'  ,
    'lPCUN',
    'lCUN' ,
    'lPCAL',
    'lLOC' ,
    'lLING',
    'lFUS' ,
    'lPARH',
    'lENT' ,
    'lTP'  ,
    'lIT'  ,
    'lMT'  ,
    'lBSTS',
    'lST'  ,
    'lTT'
]            


# define the names of the input files where the correlation coefficients were stored
TVB_RS_subj1  = 'subject_tvb/output.RestingState_01/xcorr_matrix_66_regions.npy'
TVB_RS_subj2  = 'subject_tvb/output.RestingState_02/xcorr_matrix_66_regions.npy'
TVB_RS_subj3  = 'subject_tvb/output.RestingState_03/xcorr_matrix_66_regions.npy'
TVB_RS_subj4  = 'subject_tvb/output.RestingState_04/xcorr_matrix_66_regions.npy'
TVB_RS_subj5  = 'subject_tvb/output.RestingState_05/xcorr_matrix_66_regions.npy'
TVB_RS_subj6  = 'subject_tvb/output.RestingState_06/xcorr_matrix_66_regions.npy'
TVB_RS_subj7  = 'subject_tvb/output.RestingState_07/xcorr_matrix_66_regions.npy'
TVB_RS_subj8  = 'subject_tvb/output.RestingState_08/xcorr_matrix_66_regions.npy'
TVB_RS_subj9  = 'subject_tvb/output.RestingState_09/xcorr_matrix_66_regions.npy'
TVB_RS_subj10 = 'subject_tvb/output.RestingState_10/xcorr_matrix_66_regions.npy'

TVB_LSNM_RS_subj1  = 'subject_11/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj2  = 'subject_12/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj3  = 'subject_13/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj4  = 'subject_14/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj5  = 'subject_15/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj6  = 'subject_16/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj7  = 'subject_17/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj8  = 'subject_18/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj9  = 'subject_19/output.RestingState/xcorr_matrix_66_regions.npy'
TVB_LSNM_RS_subj10 = 'subject_20/output.RestingState/xcorr_matrix_66_regions.npy'

TVB_LSNM_PV_subj1  = 'subject_11/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj2  = 'subject_12/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj3  = 'subject_13/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj4  = 'subject_14/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj5  = 'subject_15/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj6  = 'subject_16/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj7  = 'subject_17/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj8  = 'subject_18/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj9  = 'subject_19/output.PassiveViewing/xcorr_matrix_66_regions.npy'
TVB_LSNM_PV_subj10 = 'subject_20/output.PassiveViewing/xcorr_matrix_66_regions.npy'

TVB_LSNM_DMS_subj1  = 'subject_11/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj2  = 'subject_12/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj3  = 'subject_13/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj4  = 'subject_14/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj5  = 'subject_15/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj6  = 'subject_16/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj7  = 'subject_17/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj8  = 'subject_18/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj9  = 'subject_19/output.DMSTask/xcorr_matrix_66_regions.npy'
TVB_LSNM_DMS_subj10 = 'subject_20/output.DMSTask/xcorr_matrix_66_regions.npy'


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

# now, we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
tvb_rs_z        = np.arctanh(tvb_rs)
tvb_lsnm_rs_z   = np.arctanh(tvb_lsnm_rs)
tvb_lsnm_pv_z   = np.arctanh(tvb_lsnm_pv)
tvb_lsnm_dms_z  = np.arctanh(tvb_lsnm_dms)

# calculate the mean of correlation coefficients across all given subjects
tvb_rs_z_mean = np.mean(tvb_rs_z, axis=0)
tvb_lsnm_rs_z_mean = np.mean(tvb_lsnm_rs_z, axis=0)
tvb_lsnm_pv_z_mean = np.mean(tvb_lsnm_pv_z, axis=0)
tvb_lsnm_dms_z_mean = np.mean(tvb_lsnm_dms_z, axis=0)

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

# now, convert back to from Z to R correlation coefficients
tvb_rs_mean  = np.tanh(tvb_rs_z_mean)
tvb_lsnm_rs_mean  = np.tanh(tvb_lsnm_rs_z_mean)
tvb_lsnm_pv_mean  = np.tanh(tvb_lsnm_pv_z_mean)
tvb_lsnm_dms_mean  = np.tanh(tvb_lsnm_dms_z_mean)

#initialize new figure for correlations of TVB-only Resting State mean
fig = plt.figure('Mean TVB-only RS FC')
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(tvb_rs_mean, vmin=-.4, vmax=1, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

fig.savefig('mean_tvb_only_rs_fc.png')

# plot the correlation coefficients for rPC
fig = plt.figure('Correlation coefficients using rPC as seed')
ax = fig.add_subplot(111)
rPC_loc   = labels.index('rPC')
y_pos = np.arange(len(labels))
ax.barh(y_pos, tvb_rs_mean[rPC_loc])
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Correlation coefficient')

# plot the correlation coefficients for lPCUN
fig = plt.figure('Correlation coefficients using lPCUN as seed')
ax = fig.add_subplot(111)
lPCUN_loc   = labels.index('lPCUN')
y_pos = np.arange(len(labels))
ax.barh(y_pos, tvb_rs_mean[lPCUN_loc])
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel('Correlation coefficient')



# generate 1000 random matrices that will be used as a reference to compare
# functional connectivity matrices
rand_mat = np.zeros((1000, 66, 66))
for r in range(0,1000):
    rand_mat[r] = np.random.rand(66,66)
    
# threshold the connectivity matrix to preserve only a proportion 'p' of
# the strongest weights, then binarize the matrix
min_sparsity = 0.1 #0.06
max_sparsity = 0.5 #0.4
num_sparsity = 40  #35
TVB_RS_EFFICIENCY = np.zeros(num_sparsity)
RAND_MAT_EFFICIENCY=np.zeros(num_sparsity)
TVB_RS_CLUSTERING = np.zeros(num_sparsity)
RAND_MAT_CLUSTERING=np.zeros(num_sparsity)
TVB_RS_MODULARITY = np.zeros(num_sparsity)
RAND_MAT_MODULARITY=np.zeros(num_sparsity)
TVB_RS_DEGREE     = np.zeros((num_sparsity, 66))
i = 0
threshold_array = np.linspace(min_sparsity, max_sparsity, num_sparsity)
for p in threshold_array: 

    # claculate metrics for random matrices first. We are only interested in keeping an
    # average of each metric per random matrix.
    for r in range(0,1):
        rand_mat_p  = bct.threshold_proportional(rand_mat[r], p, copy=True)
        rand_mat_bin= bct.binarize(rand_mat_p, copy=True)
        rand_mat_global_efficiency = bct.efficiency_bin(rand_mat_bin, local=False)
        RAND_MAT_EFFICIENCY[i] += rand_mat_global_efficiency
        rand_mat_mean_clustering = np.mean(bct.clustering_coef_bu(rand_mat_bin))
        RAND_MAT_CLUSTERING[i] += rand_mat_mean_clustering
        rand_mat_modularity = bct.modularity_und(rand_mat_bin, gamma=1, kci=None)[1]
        RAND_MAT_MODULARITY[i] += rand_mat_modularity
    # average graph metrics at the ith sparsity level across all random matrices
    RAND_MAT_EFFICIENCY[i] = RAND_MAT_EFFICIENCY[i] / 1000.
    RAND_MAT_CLUSTERING[i] = RAND_MAT_CLUSTERING[i] / 1000.
    RAND_MAT_MODULARITY[i] = RAND_MAT_MODULARITY[i] / 1000.

    tvb_rs_mean_p   = bct.threshold_proportional(tvb_rs_mean, p, copy=True)
    tvb_rs_mean_bin = bct.binarize(tvb_rs_mean_p, copy=True)

    # calculate global efficiency using Brain Connectivity Toolbox
    tvb_rs_global_efficiency = bct.efficiency_bin(tvb_rs_mean_bin, local=False)
    TVB_RS_EFFICIENCY[i] = tvb_rs_global_efficiency

    # calculate clustering coefficient vector using Brain Connectivity Toolbox
    tvb_rs_mean_clustering_coefficient = np.mean(bct.clustering_coef_bu(tvb_rs_mean_bin))
    TVB_RS_CLUSTERING[i] = tvb_rs_mean_clustering_coefficient

    # calculate modularity using Brain Connectivity Toolbox
    tvb_rs_modularity = bct.modularity_und(tvb_rs_mean_bin, gamma=1, kci=None)
    TVB_RS_MODULARITY[i] = tvb_rs_modularity[1]

    # calculate nodal degree using BCT
    tvb_rs_nodal_degree = bct.degrees_und(tvb_rs_mean_bin)
    TVB_RS_DEGREE[i] = tvb_rs_nodal_degree

    # save unweighted undirected matrix at the given sparsity level
    # then, convert the matrix to an edge table and save that as well.
    if p == 0.2:
        df = pd.DataFrame(tvb_rs_mean_bin, index=labels, columns=labels)
        df.to_csv('tvb_rs_mean_bin.csv')
        adj_to_list('tvb_rs_mean_bin.csv', 'tvb_rs_mean_edge_table.csv', ',')

    i = i + 1
    
# calculate nodal degree using BCT
tvb_rs_nodal_degree = bct.degrees_und(tvb_rs_mean_p)

# calculate nodal strength using BCT
tvb_rs_nodal_strength = bct.strengths_und_sign(tvb_rs_mean_p)

#initialize new figure for correlations of TVB/LSNM RS FC mean
fig = plt.figure('Mean TVB/LSNM RS FC')
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(tvb_lsnm_rs_mean, vmin=-.4, vmax=1, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

fig.savefig('mean_tvb_lsnm_rs_fc.png')

# threshold the connectivity matrix to preserve only a proportion 'p' of
# the strongest weights, then binarize the matrix
TVB_LSNM_RS_EFFICIENCY = np.zeros(num_sparsity)
TVB_LSNM_RS_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_RS_MODULARITY = np.zeros(num_sparsity)
TVB_LSNM_RS_DEGREE     = np.zeros((num_sparsity, 66))
i = 0
for p in threshold_array: 

    tvb_lsnm_rs_mean_p = bct.threshold_proportional(tvb_lsnm_rs_mean, p, copy=True)
    tvb_lsnm_rs_mean_bin = bct.binarize(tvb_lsnm_rs_mean_p, copy=True)

    # calculate global efficiency using Brain Connectivity Toolbox
    tvb_lsnm_rs_global_efficiency = bct.efficiency_bin(tvb_lsnm_rs_mean_bin, local=False)
    TVB_LSNM_RS_EFFICIENCY[i] = tvb_lsnm_rs_global_efficiency

    # calculate clustering coefficient vector using Brain Connectivity Toolbox
    tvb_lsnm_rs_mean_clustering_coefficient = np.mean(bct.clustering_coef_bu(tvb_lsnm_rs_mean_bin))
    TVB_LSNM_RS_CLUSTERING[i] = tvb_lsnm_rs_mean_clustering_coefficient

    # calculate modularity using Brain Connectivity Toolbox
    tvb_lsnm_rs_modularity = bct.modularity_und(tvb_lsnm_rs_mean_bin, gamma=1, kci=None)
    TVB_LSNM_RS_MODULARITY[i] = tvb_lsnm_rs_modularity[1]

    # calculate nodal degree using BCT
    tvb_lsnm_rs_nodal_degree = bct.degrees_und(tvb_lsnm_rs_mean_bin)
    TVB_LSNM_RS_DEGREE[i] = tvb_lsnm_rs_nodal_degree

    # save unweighted undirected matrix at the given sparsity level
    # then, convert the matrix to an edge table and save that as well.
    if p == 0.2:
        df = pd.DataFrame(tvb_lsnm_rs_mean_bin, index=labels, columns=labels)
        df.to_csv('tvb_lsnm_rs_mean_bin.csv')
        adj_to_list('tvb_lsnm_rs_mean_bin.csv', 'tvb_lsnm_rs_mean_edge_table.csv', ',')

    i = i + 1
    
# calculate nodal degree using BCT
tvb_lsnm_rs_nodal_degree = bct.degrees_und(tvb_lsnm_rs_mean_p)

# calculate nodal strength using BCT
tvb_lsnm_rs_nodal_strength = bct.strengths_und_sign(tvb_lsnm_rs_mean_p)

#initialize new figure for correlations of TVB/LSNM PV FC mean
fig = plt.figure('Mean TVB/LSNM PV FC')
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(tvb_lsnm_pv_mean, vmin=-.4, vmax=1, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

fig.savefig('mean_tvb_lsnm_pv_fc.png')

# threshold the connectivity matrix to preserve only a proportion 'p' of
# the strongest weights, then binarize the matrix
TVB_LSNM_PV_EFFICIENCY = np.zeros(num_sparsity)
TVB_LSNM_PV_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_PV_MODULARITY = np.zeros(num_sparsity)
TVB_LSNM_PV_DEGREE     = np.zeros((num_sparsity, 66))
i = 0
for p in threshold_array: 

    tvb_lsnm_pv_mean_p = bct.threshold_proportional(tvb_lsnm_pv_mean, p, copy=True)
    tvb_lsnm_pv_mean_bin = bct.binarize(tvb_lsnm_pv_mean_p, copy=True)
    
    # calculate global efficiency using Brain Connectivity Toolbox
    tvb_lsnm_pv_global_efficiency = bct.efficiency_bin(tvb_lsnm_pv_mean_bin, local=False)
    TVB_LSNM_PV_EFFICIENCY[i] = tvb_lsnm_pv_global_efficiency

    # calculate clustering coefficient vector using Brain Connectivity Toolbox
    tvb_lsnm_pv_mean_clustering_coefficient = np.mean(bct.clustering_coef_bu(tvb_lsnm_pv_mean_bin))
    TVB_LSNM_PV_CLUSTERING[i] = tvb_lsnm_pv_mean_clustering_coefficient

    # calculate modularity using Brain Connectivity Toolbox
    tvb_lsnm_pv_modularity = bct.modularity_und(tvb_lsnm_pv_mean_bin, gamma=1, kci=None)
    TVB_LSNM_PV_MODULARITY[i] = tvb_lsnm_pv_modularity[1]

    # calculate nodal degree using BCT
    tvb_lsnm_pv_nodal_degree = bct.degrees_und(tvb_lsnm_pv_mean_bin)
    TVB_LSNM_PV_DEGREE[i] = tvb_lsnm_pv_nodal_degree

    # save unweighted undirected matrix at the given sparsity level
    # then, convert the matrix to an edge table and save that as well.
    if p == 0.2:
        df = pd.DataFrame(tvb_lsnm_pv_mean_bin, index=labels, columns=labels)
        df.to_csv('tvb_lsnm_pv_mean_bin.csv')
        adj_to_list('tvb_lsnm_pv_mean_bin.csv', 'tvb_lsnm_pv_mean_edge_table.csv', ',')

    i = i+1
    
# calculate nodal degree using BCT
tvb_lsnm_pv_nodal_degree = bct.degrees_und(tvb_lsnm_pv_mean_p)

# calculate nodal strength using BCT
tvb_lsnm_pv_nodal_strength = bct.strengths_und_sign(tvb_lsnm_pv_mean_p)

#initialize new figure for correlations of TVB/LSNM DMS FC mean
fig = plt.figure('Mean TVB/LSNM DMS FC')
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(tvb_lsnm_dms_mean, vmin=-.4, vmax=1, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

fig.savefig('mean_tvb_lsnm_dms_fc.png')

# threshold the connectivity matrix to preserve only a proportion 'p' of
# the strongest weights, then binarize the matrix
TVB_LSNM_DMS_EFFICIENCY = np.zeros(num_sparsity)
TVB_LSNM_DMS_CLUSTERING = np.zeros(num_sparsity)
TVB_LSNM_DMS_MODULARITY = np.zeros(num_sparsity)
TVB_LSNM_DMS_DEGREE     = np.zeros((num_sparsity, 66))
i = 0
for p in threshold_array: 

    tvb_lsnm_dms_mean_p = bct.threshold_proportional(tvb_lsnm_dms_mean, p, copy=True)
    tvb_lsnm_dms_mean_bin = bct.binarize(tvb_lsnm_dms_mean_p, copy=True)

    # calculate global efficiency using Brain Connectivity Toolbox
    tvb_lsnm_dms_global_efficiency = bct.efficiency_bin(tvb_lsnm_dms_mean_bin, local=False)
    TVB_LSNM_DMS_EFFICIENCY[i] = tvb_lsnm_dms_global_efficiency

    # calculate clustering coefficient vector using Brain Connectivity Toolbox
    tvb_lsnm_dms_clustering_coefficient = np.mean(bct.clustering_coef_bu(tvb_lsnm_dms_mean_bin))
    TVB_LSNM_DMS_CLUSTERING[i] = tvb_lsnm_dms_clustering_coefficient

    # calculate modularity using Brain Connectivity Toolbox
    tvb_lsnm_dms_modularity = bct.modularity_und(tvb_lsnm_dms_mean_bin, gamma=1, kci=None)
    TVB_LSNM_DMS_MODULARITY[i] = tvb_lsnm_dms_modularity[1]

    # calculate nodal degree using BCT
    tvb_lsnm_dms_nodal_degree = bct.degrees_und(tvb_lsnm_dms_mean_bin)
    TVB_LSNM_DMS_DEGREE[i] = tvb_lsnm_dms_nodal_degree

    # save unweighted undirected matrix at the given sparsity level
    # then, convert the matrix to an edge table and save that as well.
    if p == 0.2:
        df = pd.DataFrame(tvb_lsnm_dms_mean_bin, index=labels, columns=labels)
        df.to_csv('tvb_lsnm_dms_mean_bin.csv')
        adj_to_list('tvb_lsnm_dms_mean_bin.csv', 'tvb_lsnm_dms_mean_edge_table.csv', ',')

    i = i + 1

# initialize new figures for measures plot at different sparsities
fig = plt.figure('Global Efficiency')
cax = fig.add_subplot(111)
plt.plot(threshold_array, TVB_RS_EFFICIENCY, label='TVB RS')
plt.plot(threshold_array, TVB_LSNM_RS_EFFICIENCY, label='TVB/LSNM RS')
plt.plot(threshold_array, TVB_LSNM_PV_EFFICIENCY, label='TVB/LSNM PV')
plt.plot(threshold_array, TVB_LSNM_DMS_EFFICIENCY, label='TVB/LSNM DMS')
plt.plot(threshold_array, RAND_MAT_EFFICIENCY, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Global Efficiency')
plt.legend(loc='lower right')
fig.savefig('efficiency_across_thresholds.png')

fig = plt.figure('Mean Clustering')
cax = fig.add_subplot(111)
plt.plot(threshold_array, TVB_RS_CLUSTERING, label='TVB RS')
plt.plot(threshold_array, TVB_LSNM_RS_CLUSTERING, label='TVB/LSNM RS')
plt.plot(threshold_array, TVB_LSNM_PV_CLUSTERING, label='TVB/LSNM PV')
plt.plot(threshold_array, TVB_LSNM_DMS_CLUSTERING, label='TVB/LSNM DMS')
plt.plot(threshold_array, RAND_MAT_CLUSTERING, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Mean Clustering')
plt.legend(loc='lower right')
fig.savefig('clustering_across_thresholds.png')

fig = plt.figure('Modularity Index')
cax = fig.add_subplot(111)
plt.plot(threshold_array, TVB_RS_MODULARITY, label='TVB RS')
plt.plot(threshold_array, TVB_LSNM_RS_MODULARITY, label='TVB/LSNM RS')
plt.plot(threshold_array, TVB_LSNM_PV_MODULARITY, label='TVB/LSNM PV')
plt.plot(threshold_array, TVB_LSNM_DMS_MODULARITY, label='TVB/LSNM DMS')
plt.plot(threshold_array, RAND_MAT_MODULARITY, label='Random')
plt.xlabel('Threshold')
plt.ylabel('Modularity Index')
plt.legend(loc='upper right')
fig.savefig('modularity_across_thresholds.png')

# calculate nodal degree using BCT
tvb_lsnm_dms_nodal_degree = bct.degrees_und(tvb_lsnm_dms_mean_p)

# calculate nodal strength using BCT
tvb_lsnm_dms_nodal_strength = bct.strengths_und_sign(tvb_lsnm_dms_mean_p)

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
index = np.arange(66)
bar_width = 1
colors = 'lightblue '*66            # create array of colors for bar chart
c_tvb_rs_s = colors.split()         # one for each bar chart type
c_tvb_lsnm_rs_s = colors.split()
c_tvb_lsnm_pv_s = colors.split()
c_tvb_lsnm_dms_s = colors.split()
c_tvb_rs_k = colors.split()
c_tvb_lsnm_rs_k = colors.split()
c_tvb_lsnm_pv_k = colors.split()
c_tvb_lsnm_dms_k = colors.split()

# Find 10 maximum values for each condition and for each metric,
# then highlight top 10 by changing bar color to red
top_10_s1 = tvb_rs_nodal_strength[0].argsort()[-10:][::-1]
for idx in top_10_s1:
    c_tvb_rs_s[idx] = 'red'
top_10_s2 = tvb_lsnm_rs_nodal_strength[0].argsort()[-10:][::-1]
for idx in top_10_s2:
    c_tvb_lsnm_rs_s[idx] = 'red'
top_10_s3 = tvb_lsnm_pv_nodal_strength[0].argsort()[-10:][::-1]
for idx in top_10_s3:
    c_tvb_lsnm_pv_s[idx] = 'red'
top_10_s4 = tvb_lsnm_dms_nodal_strength[0].argsort()[-10:][::-1]
for idx in top_10_s4:
    c_tvb_lsnm_dms_s[idx] = 'red'
top_10_k1 = tvb_rs_nodal_degree.argsort()[-10:][::-1]
for idx in top_10_k1:
    c_tvb_rs_k[idx] = 'red'
top_10_k2 = tvb_lsnm_rs_nodal_degree.argsort()[-10:][::-1]
for idx in top_10_k2:
    c_tvb_lsnm_rs_k[idx] = 'red'
top_10_k3 = tvb_lsnm_pv_nodal_degree.argsort()[-10:][::-1]
for idx in top_10_k3:
    c_tvb_lsnm_pv_k[idx] = 'red'
top_10_k4 = tvb_lsnm_dms_nodal_degree.argsort()[-10:][::-1]
for idx in top_10_k4:
    c_tvb_lsnm_dms_k[idx] = 'red'
    
# generate bar charts
f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
ax1.bar(index, tvb_rs_nodal_strength[0], bar_width, color=c_tvb_rs_s)
ax1.set_title('Nodal strength for all 66 nodes')
ax2.bar(index, tvb_lsnm_rs_nodal_strength[0], bar_width, color=c_tvb_lsnm_rs_s)
ax3.bar(index, tvb_lsnm_pv_nodal_strength[0], bar_width, color=c_tvb_lsnm_pv_s)
ax4.bar(index, tvb_lsnm_dms_nodal_strength[0], bar_width, color=c_tvb_lsnm_dms_s)

plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xticks(index + bar_width/2.0, labels, rotation='vertical')

# initialize new figure for nodal degree bar charts
f, (ax5, ax6, ax7, ax8) = plt.subplots(4, sharex=True, sharey=True)
ax5.bar(index, tvb_rs_nodal_degree, bar_width, color=c_tvb_rs_k)
ax5.set_title('Nodal degree for all 66 nodes')
ax6.bar(index, tvb_lsnm_rs_nodal_degree, bar_width, color=c_tvb_lsnm_rs_k)
ax7.bar(index, tvb_lsnm_pv_nodal_degree, bar_width, color=c_tvb_lsnm_pv_k)
ax8.bar(index, tvb_lsnm_dms_nodal_degree, bar_width, color=c_tvb_lsnm_dms_k)

plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xticks(index + bar_width/2.0, labels, rotation='vertical')


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
fig = plt.figure('TVB Resting State')
cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_rs_mean.shape[0], k=0)
mask = np.transpose(mask)
tvb_rs_mean = np.ma.array(tvb_rs_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
corr_mat_tvb_rs = np.ma.ravel(tvb_rs_mean)

# remove masked elements from cross-correlation matrix
corr_mat_tvb_rs = np.ma.compressed(corr_mat_tvb_rs)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat_tvb_rs, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 600])

fig.savefig('tvb_rs_hist_66_ROIs')


# calculate and print kurtosis
print '\nTVB Resting-State Fishers kurtosis: ', kurtosis(corr_mat_tvb_rs, fisher=True)
print 'TVB Resting-State Skewness: ', skew(corr_mat_tvb_rs)

# initialize new figure for tvb/lsnm resting-state histogram
fig = plt.figure('TVB/LSNM Resting State')
cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_lsnm_rs_mean.shape[0], k=0)
mask = np.transpose(mask)
tvb_lsnm_rs_mean = np.ma.array(tvb_lsnm_rs_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
corr_mat_tvb_lsnm_rs = np.ma.ravel(tvb_lsnm_rs_mean)

# remove masked elements from cross-correlation matrix
corr_mat_tvb_lsnm_rs = np.ma.compressed(corr_mat_tvb_lsnm_rs)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat_tvb_lsnm_rs, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 600])

fig.savefig('tvb_lsnm_rs_hist_66_ROIs')

# calculate and print kurtosis
print '\nTVB/LSNM Resting-State Fishers kurtosis: ', kurtosis(corr_mat_tvb_lsnm_rs, fisher=True)
print 'TVB/LSNM Resting-State Skewness: ', skew(corr_mat_tvb_lsnm_rs)

# initialize new figure for tvb/lsnm resting-state histogram
fig = plt.figure('TVB/LSNM Passive Viewing')
cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_lsnm_pv_mean.shape[0], k=0)
mask = np.transpose(mask)
tvb_lsnm_pv_mean = np.ma.array(tvb_lsnm_pv_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
corr_mat_tvb_lsnm_pv = np.ma.ravel(tvb_lsnm_pv_mean)

# remove masked elements from cross-correlation matrix
corr_mat_tvb_lsnm_pv = np.ma.compressed(corr_mat_tvb_lsnm_pv)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat_tvb_lsnm_pv, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 600])

fig.savefig('tvb_lsnm_pv_hist_66_ROIs')

# calculate and print kurtosis
print '\nTVB/LSNM Passive Viewing Fishers kurtosis: ', kurtosis(corr_mat_tvb_lsnm_pv, fisher=True)
print 'TVB/LSNM Passive Viewing Skewness: ', skew(corr_mat_tvb_lsnm_pv)

# initialize new figure for tvb/lsnm resting-state histogram
fig = plt.figure('TVB/LSNM DMS Task')
cax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_lsnm_dms_mean.shape[0], k=0)
mask = np.transpose(mask)
tvb_lsnm_dms_mean = np.ma.array(tvb_lsnm_dms_mean, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
corr_mat_tvb_lsnm_dms = np.ma.ravel(tvb_lsnm_dms_mean)

# remove masked elements from cross-correlation matrix
corr_mat_tvb_lsnm_dms = np.ma.compressed(corr_mat_tvb_lsnm_dms)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat_tvb_lsnm_dms, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 600])

fig.savefig('tvb_lsnm_dms_hist_66_ROIs')

# calculate and print kurtosis
print '\nTVB/LSNM DMS Task Fishers kurtosis: ', kurtosis(corr_mat_tvb_lsnm_dms, fisher=True)
print 'TVB/LSNM DMS Task Skewness: ', skew(corr_mat_tvb_lsnm_dms)

# find the 10 highest correlation values in a cross-correlation matrix
#corr_mat_tvb_lsnm_dms_sorted = np.sort(corr_mat_tvb_lsnm_dms, axis=None)[::-1]
#np.set_printoptions(threshold='nan')
#print 'Sorted FCs in DMS mean array', corr_mat_tvb_lsnm_dms_sorted[:20]

# plot scatter plots to show correlations of TVB-RS vs TVB-LSNM-RS,
# TVB-LSNM RS vs TVB-LSNM PV, and TVB-LSNM PV vs TVB-LSNM DMS 

# start with a rectangular Figure
fig=plt.figure('TVB-Only RS FC v TVB/LSNM RS FC')

# should we convert to Fisher's Z values prior to computing correlation of FCs?
#tvb_rs_z        = np.arctanh(corr_mat_tvb_rs)
#tvb_lsnm_rs_z   = np.arctanh(corr_mat_tvb_lsnm_rs)
#tvb_lsnm_pv_z   = np.arctanh(corr_mat_tvb_lsnm_pv)
#tvb_lsnm_dms_z  = np.arctanh(corr_mat_tvb_lsnm_dm)

# scatter plot
plt.scatter(corr_mat_tvb_rs, corr_mat_tvb_lsnm_rs)
plt.xlabel('TVB-Only RS FC')
plt.ylabel('TVB/LSNM RS FC')
plt.axis([-0.5,0.5,-0.5,1])

# fit scatter plot with np.polyfit
m, b = np.polyfit(corr_mat_tvb_rs, corr_mat_tvb_lsnm_rs, 1)
plt.plot(corr_mat_tvb_rs, m*corr_mat_tvb_rs + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(corr_mat_tvb_rs, corr_mat_tvb_lsnm_rs)[1,0]
plt.text(-0.4, 0.75, 'r=' + '{:.2f}'.format(r))

fig.savefig('rs_vs_rs_scatter_66_ROIs')

# start with a rectangular Figure
fig=plt.figure('TVB/LSNM RS FC v TVB/LSNM PV FC')

# scatter plot
plt.scatter(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_pv)
plt.xlabel('TVB/LSNM RS FC')
plt.ylabel('TVB/LSNM PV FC')
plt.axis([-0.5,0.5,-0.5,1])

# fit scatter plot with np.polyfit
m, b = np.polyfit(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_pv, 1)
plt.plot(corr_mat_tvb_lsnm_rs, m*corr_mat_tvb_lsnm_rs + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_pv)[1,0]
plt.text(-0.4, 0.75, 'r=' + '{:.2f}'.format(r))

fig.savefig('rs_vs_pv_scatter_66_ROIs')

# start with a rectangular Figure
fig=plt.figure('TVB/LSNM RS FC v TVB/LSNM DMS FC')

# scatter plot
plt.scatter(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_dms)  
plt.xlabel('TVB/LSNM RS FC')
plt.ylabel('TVB/LSNM DMS FC')
plt.axis([-0.5,0.5,-0.5,1])

# fit scatter plot with np.polyfit
m, b = np.polyfit(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_dms, 1)
plt.plot(corr_mat_tvb_lsnm_rs, m*corr_mat_tvb_lsnm_rs + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(corr_mat_tvb_lsnm_rs, corr_mat_tvb_lsnm_dms)[1,0]
plt.text(-0.4, 0.75, 'r=' + '{:.2f}'.format(r))

fig.savefig('rs_vs_dms_scatter_66_ROIs')

# initialize figure to plot correlations between degree of RS v PV and RS v DMS
fig=plt.figure('Degree correlation between TVB/LSNM RS PV DMS at various sparsities')

# compute correlation coefficients for all sparsities and conditions of interest
r_array_RS_v_PV = np.zeros(40)
r_array_RS_v_DMS = np.zeros(40)
r_array_PV_v_DMS = np.zeros(40)
i=0
for p in threshold_array:
    r_array_RS_v_PV[i] = np.corrcoef(TVB_LSNM_RS_DEGREE[i], TVB_LSNM_PV_DEGREE[i])[1,0]
    r_array_RS_v_DMS[i] = np.corrcoef(TVB_LSNM_RS_DEGREE[i], TVB_LSNM_DMS_DEGREE[i])[1,0]
    r_array_PV_v_DMS[i] = np.corrcoef(TVB_LSNM_PV_DEGREE[i], TVB_LSNM_DMS_DEGREE[i])[1,0]
    i = i+1

plt.plot(threshold_array, r_array_RS_v_PV,  label='TVB/LSNM RS v PV')
plt.plot(threshold_array, r_array_RS_v_DMS, label='TVB/LSNM RS v DMS')
plt.plot(threshold_array, r_array_PV_v_DMS, label='TVB/LSNM PV v DMS')
plt.legend(loc='best')
plt.xlabel('Correlation')
plt.ylabel('Sparsity')

fig.savefig('corr_RS_PV_DMS.png')

# initialize figure to plot degree correlations at sparsity 10%
fig=plt.figure('Degree Correlation between TVB/LSNM RS and TVB/LSNM PV AT 10%')

plt.scatter(TVB_LSNM_RS_DEGREE[0], TVB_LSNM_PV_DEGREE[0])
plt.xlabel('TVB/LSNM RS DEGREE')
plt.ylabel('TVB/LSNM PV DEGREE')

# fit scatter plot with np.polyfit
m, b = np.polyfit(TVB_LSNM_RS_DEGREE[0], TVB_LSNM_PV_DEGREE[0], 1)
plt.plot(TVB_LSNM_RS_DEGREE[0], m*TVB_LSNM_RS_DEGREE[0] + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(TVB_LSNM_RS_DEGREE[0], TVB_LSNM_PV_DEGREE[0])[1,0]
plt.text(1, 22, 'r=' + '{:.2f}'.format(r))

# initialize figure to plot degree correlations at sparsity 10%
fig=plt.figure('Degree Correlation between TVB/LSNM RS and TVB/LSNM DMS AT 10%')

plt.scatter(TVB_LSNM_RS_DEGREE[0], TVB_LSNM_DMS_DEGREE[0])
plt.xlabel('TVB/LSNM RS DEGREE')
plt.ylabel('TVB/LSNM DMS DEGREE')

# fit scatter plot with np.polyfit
m, b = np.polyfit(TVB_LSNM_RS_DEGREE[0], TVB_LSNM_DMS_DEGREE[0], 1)
plt.plot(TVB_LSNM_RS_DEGREE[0], m*TVB_LSNM_RS_DEGREE[0] + b, '-', color='red')

# calculate correlation coefficient and display it on plot
r = np.corrcoef(TVB_LSNM_RS_DEGREE[0], TVB_LSNM_DMS_DEGREE[0])[1,0]
plt.text(1, 22, 'r=' + '{:.2f}'.format(r))


# Show the plots on the screen
plt.show()
