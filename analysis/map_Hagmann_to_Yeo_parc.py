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
#   This file (avg_FC_TVB_ROIs_across_subjects.py) was created on June 6 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on June 8 2017
#
#
# Python function to create a map from Hagmann's 998-node connectome to Yeo's
# 7 module parcellation
#
# Uses 163,842 vertices provided by Freesurfer (fsaverage)
# and Tal-to-MNI coordinates by GingerAle
#
# **************************************************************************/
#
# map_Hagmann_to_Yeo_parc.py
#
# 1) Opens two text files (RH and LH) that contain 163,842 vertices from
# freesurfer's fsaverage, opens Yeo functional parcellation files (RH and LH)
# and put together arrays to pair each vertex with one of 7 brain modules
# as defined by Yeo et al (2011).
#
# 2) Opens a text file that contains the 998 nodes belonging to Hagmann's connectome,
# and, for each Hagmann node, find the closest coordinate in fsaverage's vertices array.
#
# 3) Assigns one of the 7 functional brain modules to each one of the 998 nodes
# in Hagmann's connectome, then save it in an npy file for later use.
#
# 4) Finally, displays Hagmann nodes on transparent brain, color coding each
# node to signal membership in one of 7 brain regions (Yeo et al 2011)

from tvb.simulator.lab import *

import os
from os.path import join as pjoin
from surfer import Brain

from mayavi import mlab

import numpy as np
import nibabel as nib

from scipy.spatial import distance


# number of vertices in each hemisphere's surface
num_vert = 163842

# open files (huge) that contain surface for RH and LH in (freesurfer's fsaverage)
rh_surf = np.loadtxt('rh.pial.asc', usecols=(0, 1, 2))
lh_surf = np.loadtxt('lh.pial.asc', usecols=(0, 1, 2))

# Load Hagmann's brain nodes
hagmann_nodes = np.loadtxt('hagmann_nodes_tal.txt')

# extract coordinates of all vertices 
rh_vert = rh_surf[0:num_vert]
lh_vert = lh_surf[0:num_vert]

# declare variables for freesurfer's fsaverage
subject_id = "fsaverage"
subjects_dir = os.environ["SUBJECTS_DIR"]

# read from Yeo parcellation's mapping file (RH and LH separately)
annot_path_rh = pjoin(subjects_dir, subject_id, 'label', 'rh.Yeo2011_7Networks_N1000.annot')
annot_path_lh = pjoin(subjects_dir, subject_id, 'label', 'lh.Yeo2011_7Networks_N1000.annot')

labels_rh, ctab_rh, names_rh = nib.freesurfer.read_annot(annot_path_rh)
labels_lh, ctab_lh, names_lh = nib.freesurfer.read_annot(annot_path_lh)

# get rid of all vertices with label zero (medial wall) as we are not interested in finding membership
# of hagmann nodes to the "median wall"
rh_vert = rh_vert[labels_rh != 0]           # get rid of vertices labeled as zero (RH)
labels_rh = labels_rh[labels_rh != 0]     # get rid of zero labels from labels array (RH)
lh_vert = lh_vert[labels_lh != 0]           # get rid of vertices labeled as zero (LH) 
labels_lh = labels_lh[labels_lh != 0]       # get rid of zero labels from labels array (LH)


print 'Size of labels array (RH) is: ', labels_rh.shape
print 'ctab (RH): ', ctab_rh
print 'Names (RH): ', names_rh
print 'Size of labels array (LH) is: ', labels_lh.shape
print 'ctab (LH): ', ctab_lh
print 'Names (LH): ', names_lh

print 'Size of vertex array (RH and LH)', rh_vert.shape, lh_vert.shape

print 'Hagmann nodes array (size): ', hagmann_nodes.shape

print 'First node: ', hagmann_nodes[0]

print 'First vertex: ', rh_vert[0]

# initialize a labels array for Hagmann nodes to store membership in Yeo's parcellation
hagmann_labels = np.zeros(hagmann_nodes.shape[0])

# finds distance between hagmann nodes and all vertices
print 'Calculating minimum distances (wait)...'
i = 0
for node in hagmann_nodes:

    
    if node[0] < 0 :                   # if x coordinate is negative, look into LH
        
        dist = distance.cdist([node], lh_vert, 'euclidean')
        idx = dist[0].argmin()
        hagmann_labels[i] = labels_lh[idx]

    else:                              # if y coordinate is positive, must be RH

        dist = distance.cdist([node], rh_vert, 'euclidean')
        idx = dist[0].argmin()
        hagmann_labels[i] = labels_rh[idx]

    i = i + 1

# convert to array of integers
hagmann_labels_int = hagmann_labels.astype(int)
    
# save hagmann_labels in file
np.save('hagmann_Yeo_parc_labels.npy', hagmann_labels_int)

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
centres = white_matter.centres

# Load one of the cortex 3d surface from TVB data files
CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")

plot_surface(CORTEX, op=0.08)

# Plot the 998 nodes of Hagmann's brain (uncomment if needed for visualization
# purposes

# first, declare colors that correspond to Yeo Parcellation's regions
colors = [(0., 0., 0.),           # medial wall (not really a region)
          (.5, 0., 1.),           # purple, visual
          (0., 0., 1.),           # blue, somatomotor
          (0., 1., 0.),           # green, dorsal attention
          (1., 0., 1.),           # violet, ventral attention
          (1., .8, .8),           # cream, limbic
          (1., .5, 0.),           # orange, frontoparietal
          (1., 0., 0.)]           # red, default

i = 0
for node in centres:

    mlab.points3d(node[0], 
                  node[1], 
                  node[2],
                  color=colors[hagmann_labels_int[i]],
                  scale_factor = 2.)
    i = i + 1

# Finally, show everything on screen
mlab.show(stop=True)
