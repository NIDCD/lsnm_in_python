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
#   This file (average_graph_metrics.py) was created on October 30 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on December 30 2017
#
# **************************************************************************/
#
# average_graph_metrics.py
#
# Calculates and plots mean and std of graph metrics corresponding to a
# number of simulated subjects for three conditions (PF, PV, DMS). It also
# calculates and plots mean and std of relative errors between PF and PV and
# between PV and DMS conditions.

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

import math as m

from matplotlib import cm as CM

import pandas as pd

import seaborn as sns

# declare number of graph metrics
num_of_graphs = 8

# declare number of density thresholds
num_of_densities = 8

# declare number of subjects
num_of_subs = 6

# declare number of conditions
num_of_conds = 3

# for ploting purpurses, declares density parameters
min_sparsity = 0.05
max_sparsity = 0.4
num_sparsity = num_of_densities
threshold_array = np.linspace(min_sparsity, max_sparsity, num_sparsity)

# define the names of the input files where the correlation coefficients were stored
TVB_LSNM_PF = ['subject_11/output.Fixation_dot_incl_PreSMA_3.0_0.15/graph_metrics_wei.npy',
               'subject_12/output.Fixation_dot_incl_PreSMA_3.0_0.15/graph_metrics_wei.npy',
               'subject_13/output.Fixation_dot_incl_PreSMA_3.0_0.15/graph_metrics_wei.npy',
               'subject_14/output.Fixation_dot_incl_PreSMA_3.0_0.15/graph_metrics_wei.npy',
               'subject_15/output.Fixation_dot_incl_PreSMA_3.0_0.15/graph_metrics_wei.npy',
               'subject_16/output.Fixation_dot_incl_PreSMA_3.0_0.15/graph_metrics_wei.npy']

TVB_LSNM_PV = ['subject_11/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
               'subject_12/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
               'subject_13/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
               'subject_14/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
               'subject_15/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
               'subject_16/output.PV_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy']

TVB_LSNM_DMS = ['subject_11/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
                'subject_12/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
                'subject_13/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
                'subject_14/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
                'subject_15/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy',
                'subject_16/output.DMSTask_incl_PreSMA_w_Fixation_3.0_0.15/graph_metrics_wei.npy']

#################################################################################
# Open all of the graph metrics for all subjects and conditions into numpy arrays
#################################################################################
tvb_lsnm_pf  = np.zeros((num_of_subs, num_of_graphs, num_of_densities))
tvb_lsnm_pv  = np.zeros((num_of_subs, num_of_graphs, num_of_densities))
tvb_lsnm_dms = np.zeros((num_of_subs, num_of_graphs, num_of_densities))
idx = 0
for subject in TVB_LSNM_PF:
    tvb_lsnm_pf[idx]  = np.load(subject)
    idx = idx + 1

idx = 0
for subject in TVB_LSNM_PV:
    tvb_lsnm_pv[idx]  = np.load(subject)
    idx = idx + 1

idx = 0
for subject in TVB_LSNM_DMS:
    tvb_lsnm_dms[idx] = np.load(subject)
    idx = idx + 1

################################################################################
# Calculate average and standard deviation of each graph metric per condition
################################################################################
pf_avg = np.mean(tvb_lsnm_pf,  axis=0)
pv_avg = np.mean(tvb_lsnm_pv,  axis=0)
dms_avg= np.mean(tvb_lsnm_dms, axis=0)

pf_std = np.std(tvb_lsnm_pf,  axis=0)
pv_std = np.std(tvb_lsnm_pv,  axis=0)
dms_std= np.std(tvb_lsnm_dms, axis=0)

##############################################################################
# Calculate relative error in graph metrics between all conditions and control
# from Lee et al, 2017, pp 727
##############################################################################
RE     = np.zeros((num_of_conds-1, num_of_subs, num_of_graphs))
avg_RE = np.zeros((num_of_conds-1, num_of_graphs))
std_RE = np.zeros((num_of_conds-1, num_of_graphs))

# calculate relative error between PV and PF for each subject
for idx in range(num_of_subs):
    for idx2 in range(num_of_graphs):
        RE[0, idx, idx2] = m.sqrt(np.sum(np.power(tvb_lsnm_pv[idx, idx2] - tvb_lsnm_pf[idx, idx2], 2)) / 
                                  np.sum(np.power(tvb_lsnm_pf[idx, idx2], 2)))

# calculate relative error beween DMS and PF for each subject
for idx in range(num_of_subs):
    for idx2 in range(num_of_graphs):
        RE[1, idx, idx2] = m.sqrt(np.sum(np.power(tvb_lsnm_dms[idx, idx2] - tvb_lsnm_pf[idx, idx2], 2)) / 
                                  np.sum(np.power(tvb_lsnm_pf[idx, idx2], 2)))

# average relative error across subjects and convert to %
avg_RE[0] = np.mean(RE[0], axis=0) * 100.
avg_RE[1] = np.mean(RE[1], axis=0) * 100.
std_RE[0] = np.std(RE[0],  axis=0) * 100.
std_RE[1] = np.std(RE[1],  axis=0) * 100.

print 'Array of mean differences between PV and PF: ', avg_RE[0]
print 'Array of stds between PV and PF: ', std_RE[0]
print 'Array of mean differences between DMS and PF: ', avg_RE[1]
print 'Array of stds between DMS and PF: ', std_RE[1]


####################################################################################
# Display graph theory metrics of RS, PV, DMS for all sparsity thresholds
####################################################################################
# increase font size
plt.rcParams.update({'font.size': 15})

fig = plt.figure('Global Efficiency')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[0],  pf_std[0],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[0],  pv_std[0],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[0], dms_std[0], marker='o', label='DMS')
plt.xlabel('Threshold')
plt.ylabel('Mean Global Efficiency')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_global_efficiency_across_thresholds.png')

fig = plt.figure('Mean Local Efficiency')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[1],  pf_std[1],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[1],  pv_std[1],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[1], dms_std[1], marker='o', label='DMS')
plt.xlabel('Density threshold')
plt.ylabel('Mean Local Efficiency')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_mean_local_efficiencies_across_densities.png')

fig = plt.figure('Mean Clustering Coefficient')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[2],  pf_std[2],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[2],  pv_std[2],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[2], dms_std[2], marker='o', label='DMS')
plt.xlabel('Threshold')
plt.ylabel('Mean Clustering')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_clustering_across_thresholds.png')

fig = plt.figure('Characteristic path length of a range of densities')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[3],  pf_std[3],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[3],  pv_std[3],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[3], dms_std[3], marker='o', label='DMS')
plt.xlabel('Threshold')
plt.ylabel('Characteristic Path Length')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_charpath_across_thresholds.png')

fig = plt.figure('Average Eigenvector Centrality for a range of densities')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[4],  pf_std[4],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[4],  pv_std[4],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[4], dms_std[4], marker='o', label='DMS')
plt.xlabel('Threshold')
plt.ylabel('Average Eigenvector Centrality')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_eigen_centrality_across_thresholds.png')

fig = plt.figure('Average Betweennes Centrality for a range of densities')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[5],  pf_std[5],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[5],  pv_std[5],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[5], dms_std[5], marker='o', label='DMS')
plt.xlabel('Threshold')
plt.ylabel('Average Betweennes Centrality')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_btwn_centrality_across_thresholds.png')

fig = plt.figure('Participation Coefficient')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[6],  pf_std[6],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[6],  pv_std[6],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[6], dms_std[6], marker='o', label='DMS')
plt.xlabel('Density threshold')
plt.ylabel('Participation Coefficient')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_pc_across_thresholds.png')

fig = plt.figure('Modularity')
cax = fig.add_subplot(111)
plt.errorbar(threshold_array, pf_avg[7],  pf_std[7],  marker='o', label='PF')
plt.errorbar(threshold_array, pv_avg[7],  pv_std[7],  marker='o', label='PV')
plt.errorbar(threshold_array, dms_avg[7], dms_std[7], marker='o', label='DMS')
plt.xlabel('Density threshold')
plt.ylabel('Modularity')
plt.legend(loc='best')
cax.set_xlim(min_sparsity, max_sparsity)
fig.savefig('avg_pf_pv_dms_mod_across_thresholds.png')

##############################################################################
# Plot relative error in graph metrics between all conditions and control
##############################################################################
# data to plot
n_groups = num_of_graphs

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 1.0
 
rects1 = plt.bar(index, avg_RE[0], bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=std_RE[0],
                 error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),
                 label='PV-PF')
 
rects2 = plt.bar(index + bar_width, avg_RE[1], bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=std_RE[1],
                 error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2),
                 label='DMS-PF')
 
plt.xlabel('Graph metric')
plt.ylabel('Relative Error')
plt.xticks(index + bar_width, ('GE', 'LE', 'CC', 'CP', 'EC', 'BC', 'PC', 'M'))
plt.legend(loc='best')
plt.tight_layout()
fig.savefig('avg_RE.png')

##############################################################################
# try seaborn plots using above data
##############################################################################
fig, ax = plt.subplots()
df=pd.DataFrame(data = RE[1],                
                index = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
                columns = ['GE', 'LE', 'CC', 'CP', 'EC', 'BC', 'PC', 'M']
)

ax=sns.violinplot(data=df, scale='count')
ax=sns.swarmplot(data=df, color='black')


##############################################################################
# Show the plots on the screen
##############################################################################
plt.show()

