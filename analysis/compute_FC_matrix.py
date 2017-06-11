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
#   This file (compute_FC_matrix.py) was created on June 8 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on June 9 2017
#
# **************************************************************************/

# compute_FC_matrix.py
#
# Given an array of BOLD timeseries:
# 1. Rearrange BOLD timeseries to match Yeo (2011) parcellation order (7 modules)
# 2. compute and display the correlation-based functional connectivity matrix
# 3. Calculate and display average BOLD signal for each one of Yeo's networks.

import numpy as np

import matplotlib.pyplot as plt

# define the name of the input file where the BOLD timeseries are stored
BOLD_file = 'subject_12/output.DMSTask_incl_PreSMA/bold_balloon_998_regions.npy'

# define the name of the labels corresponding to Yeo's 7-network parcellation
labels_file = 'hagmann_Yeo_parc_labels.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
fc_file = 'fc_matrix_998_regions_Yeo__parc.npy'

# define Yeo parcellation networks
Yeo_parc_networks = [7,       # DMN
                     0,       # medial wall 
                     5,       # limbic
                     4,       # ventral attention
                     6,       # frontoparietal
                     3,       # dorsal attention
                     2,       # somatomotor
                     1]       # visual


# open both files, timeseries and labels
bold   = np.load(BOLD_file)
labels = np.load(labels_file)

print 'Size of BOLD file: ', bold.shape
print 'Size of labels file: ', labels.shape

# reorder bold timeseries to an ascending label order
new_bold = np.zeros(bold.shape)
idx = 0
for network in Yeo_parc_networks:

    print 'Rearranging member nodes of network ', network

    network_loc = np.where(labels==network)             # find indices of labels that match current network within labels array
    network_size= network_loc[0].size                   # calculate number of nodes in current network

    print 'Network ', network, ' contains ', network_size, ' nodes'
    
    new_bold[idx:idx+network_size] = bold[network_loc]  # insert nodes that correspond to current network into rearranged bold

    idx = idx + network_size
    
print 'Size of bold array before removing medial wall nodes: ', new_bold.shape
    
# get rid of those bold regions labeled as zero bc they correspond to medial wall (not really a brain region)
medial_wall_nodes = np.where(labels==0)
print 'Medial wall nodes: ', medial_wall_nodes
final_bold   = np.delete(new_bold, medial_wall_nodes, axis=0)
final_labels = np.delete(labels, medial_wall_nodes, axis=0)

# use the rearranged BOLD timeseries array to compute FC matrix
fc_matrix = np.corrcoef(final_bold)
print 'Number of ROIs: ', final_bold.shape
print 'Shape of correlation matrix: ', fc_matrix.shape

# fill diagonals of FC matrix with zeros
np.fill_diagonal(fc_matrix, 0)

# save FC matrix to an npy python file
np.save(fc_file, fc_matrix)

print 'Max correlation is: ', np.amax(fc_matrix)
print 'Min correlation is: ', np.amin(fc_matrix)

# plot FC matrix as a heatmap
fig = plt.figure('FC matrix')
ax = fig.add_subplot(111)
cax = ax.imshow(fc_matrix, vmin=-0.9, vmax=0.9, interpolation='nearest', cmap='bwr')
ax.grid(color='black', linestyle='-', linewidth=2)
plt.xticks([217, 285, 388, 522, 628, 812])              # change frequency of ticks to match beginning of each network
plt.yticks([217, 285, 388, 522, 628, 812])
#color_bar=plt.colorbar(cax, orientation='horizontal')

# Show the plots on the screen
plt.show()


