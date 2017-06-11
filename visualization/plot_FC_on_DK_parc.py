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
#   This file (plot_FC_on_brain.py) was created on May 27 2017.
#
#
#   Author: Antonio Ulloa. Based on Pysurfer's plot_parc_values.py
#
#   Last updated by Antonio Ulloa on May 27 2017
#
# **************************************************************************/
#
# plot_FC_on_brain.py
#
# Displays right hemisphere of PySurfer's inflated brain and superimposes spheres
# representing visual short-term memory model brain areas.

import os
from os.path import join as pjoin
from surfer import Brain

from mayavi import mlab

import numpy as np
import nibabel as nib

print(__doc__)

# declare ROI labels
Hag_labels =  np.array([' LOF',     
                        'PORB',         
                        '  FP',          
                        ' MOF',          
                        'PTRI',          
                        'POPE',          
                        ' RMF',          
                        '  SF',          
                        ' CMF',          
                        'PREC',          
                        'PARC',          
                        ' RAC',          
                        ' CAC',          
                        '  PC',          
                        'ISTC',          
                        'PSTC',          
                        'SMAR',          
                        '  SP',          
                        '  IP',          
                        'PCUN',          
                        ' CUN',          
                        'PCAL',          
                        'LOCC',          
                        'LING',          
                        ' FUS',          
                        'PARH',          
                        ' ENT',          
                        '  TP',          
                        '  IT',          
                        '  MT',          
                        'BSTS',          
                        '  ST',          
                        '  TT'])            

# mapping from Hagmann's parcellation to DK parcellation (same parcellation, difference label order)
Hag2DK_map = [33,30,12,8,34,20,26,24,18,28,14,22,0,23,3,29,25,10,5,1,4,21,15,13,9,19,11,6,7,17,31,16,2,27,32,35] 

# location of right precuneus in FC arrays
rPCUN = 19

fc_matrix = ''

subject_id = "fsaverage"
hemi = 'lh'
surface = 'pial'
view = 'lateral'
subjects_dir = os.environ["SUBJECTS_DIR"]

# load FC differences
tvb_lsnm_rs   = np.load('tvb_lsnm_rs_fc_avg.npy')
dms_minus_rs = np.load('dms_minus_rs.npy')
pv_minus_rs  = np.load('pv_minus_rs.npy')

# calculate which ROI has the max cumulative differences
FC_max_diff_pv_rs  = np.sum(np.absolute(pv_minus_rs), axis=1)
FC_max_diff_dms_rs = np.sum(np.absolute(dms_minus_rs), axis=1)
FC_max_diff_idx_pv_rs  = np.where(FC_max_diff_pv_rs  == np.amax(FC_max_diff_pv_rs))[0][0]
FC_max_diff_idx_dms_rs = np.where(FC_max_diff_dms_rs == np.amax(FC_max_diff_dms_rs))[0][0]
print 'Maximum cummulative (DMS - RS) is at row ', FC_max_diff_idx_dms_rs
print 'Maximum cummulative ( PV - RS) is at row ', FC_max_diff_idx_pv_rs

# extract FC values for each ROI associated with rPCUN
roi_data_rs_rh     = tvb_lsnm_rs[rPCUN][0:33]
roi_data_rs_lh     = tvb_lsnm_rs[rPCUN][33:66]
roi_data_dms_rs_rh =  dms_minus_rs[FC_max_diff_idx_dms_rs][0:33]
roi_data_pv_rs_rh  =  pv_minus_rs[FC_max_diff_idx_dms_rs][0:33]
roi_data_dms_rs_lh =  dms_minus_rs[FC_max_diff_idx_dms_rs][33:66]
roi_data_pv_rs_lh  =  pv_minus_rs[FC_max_diff_idx_dms_rs][33:66]

# also, calculate the maximum correlation and minimum coefficient found
FC_rs_max = np.amax(roi_data_rs_rh)
FC_rs_min = np.amin(roi_data_rs_rh)
FC_max = max([np.amax(pv_minus_rs),
              np.amax(dms_minus_rs)])
FC_min = min([np.amin(pv_minus_rs),
              np.amin(dms_minus_rs)])

print 'Max RS FC is: ', FC_rs_max
print 'Min RS FC is: ', FC_rs_min
print 'Max FC diff is: ', FC_max
print 'Min FC diff is: ', FC_min


"""
Bring up the visualization.
"""
brain = Brain(subject_id, hemi, surface, views=view, background='white', show_toolbar=True)

"""
Show Desikan - Killiany parcellation
"""
annot_path_rh = pjoin(subjects_dir, subject_id, 'label', 'rh.aparc.annot')
annot_path_lh = pjoin(subjects_dir, subject_id, 'label', 'lh.aparc.annot')

#labels, ctab, names = nib.freesurfer.read_annot(annot_path_rh)
labels, ctab, names = nib.freesurfer.read_annot(annot_path_lh)

# reorganize the RS array to match the order of the regions found in DK parcellation array
roi_data_rs_rh=np.append(roi_data_rs_rh, [0., 0., 0.])          # add 3 more elements to Hagmann's to match DK parcellation array
roi_data_rs_lh=np.append(roi_data_rs_lh, [0., 0., 0.])          # add 3 more elements to Hagmann's to match DK parcellation array
roi_data_dms_rs_rh=np.append(roi_data_dms_rs_rh, [0., 0., 0.])
roi_data_pv_rs_rh =np.append(roi_data_pv_rs_rh, [0., 0., 0.])
roi_data_dms_rs_lh=np.append(roi_data_dms_rs_lh, [0., 0., 0.])
roi_data_pv_rs_lh =np.append(roi_data_pv_rs_lh, [0., 0., 0.])
Hag_labels=np.append(Hag_labels, ['unknown', 'corpus callosum', 'insula'])   # add 3 more elements to labels as well
print 'Original Hagmann labels: ', Hag_labels
print 'Original correlation values (RH): ', roi_data_dms_rs_rh
print 'Original correlation values (LH): ', roi_data_dms_rs_lh
roi_data_rh = roi_data_rs_rh[Hag2DK_map]
roi_data_lh = roi_data_rs_lh[Hag2DK_map]
roi_data_dms_rs_rh = roi_data_dms_rs_rh[Hag2DK_map]
roi_data_dms_rs_lh = roi_data_dms_rs_lh[Hag2DK_map]
roi_data_pv_rs_rh  = roi_data_pv_rs_rh[Hag2DK_map]
roi_data_pv_rs_lh  = roi_data_pv_rs_lh[Hag2DK_map]
new_labels = Hag_labels[Hag2DK_map]

print 'Labels after reordering to match DK parcellation: ', new_labels
print 'Corresponding correlation values (RH): ', roi_data_dms_rs_rh 
print 'Corresponding correlation values (LH): ', roi_data_dms_rs_lh 

vtx_data_rh = roi_data_pv_rs_rh[labels]
vtx_data_lh = roi_data_pv_rs_lh[labels]

brain.add_data(vtx_data_lh, -0.26, 0.26, colormap='coolwarm', alpha=.75)

# Finally, show everything on screen
mlab.show(stop=True)
