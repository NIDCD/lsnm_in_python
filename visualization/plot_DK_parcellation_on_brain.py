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
#   This file (plot_DK_parcellation_on_brain.py) was created on May 26 2017.
#
#
#   Author: Antonio Ulloa. Based on Pysurfer's plot_foci.py
#
#   Last updated by Antonio Ulloa on May 27 2017
#
# **************************************************************************/
#
# plot_DK_parcellation_on_brain.py
#
# Displays right hemisphere of PySurfer's inflated brain and superimposes spheres
# representing visual short-term memory model brain areas.

import os
from os.path import join as pjoin
from surfer import Brain

from mayavi import mlab

#import numpy as np

print(__doc__)

subject_id = "fsaverage"
hemi = 'rh'
surface = 'pial'
view = 'medial'
subjects_dir = os.environ["SUBJECTS_DIR"]

"""
Bring up the visualization.
"""
brain = Brain(subject_id, hemi, surface, views=view, cortex='high_contrast', show_toolbar=True)

"""
Show Desikan - Killiany parcellation
"""
annot_path = pjoin(subjects_dir, subject_id, 'label', 'rh.aparc.annot')
brain.add_annotation(annot_path, hemi='rh', borders=False)

mlab.savefig('parcelated_rh_medial.png')

# Finally, show everything on screen
mlab.show(stop=True)
