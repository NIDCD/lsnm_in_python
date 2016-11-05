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
#   This file (plot_auditory_nodes_on_inflated_brain.py) was created on November 2 2016.
#
#
#   Author: Antonio Ulloa. Based on Pysurfer's plot_foci.py
#
#   Last updated by Antonio Ulloa on June 16, 2015
#
# **************************************************************************/
#
# plot_auditory_nodes_on_inflated_brain.py
#
# Displays right hemisphere of PySurfer's inflated brain and superimposes spheres
# representing auditory short-term memory model brain areas.

import os
import nibabel as nib
from surfer import Brain
from surfer import utils

from mayavi import mlab

import numpy as np

print(__doc__)

subject_id = "fsaverage"
subjects_dir = os.environ["SUBJECTS_DIR"]

# Define the hypothetical Talairach locations of each LSNM auditory modules
a1 = [51,-24,8]
a2 = [61,-36,12]
# Ignore the ST posterior as we are especially interested in ST anterior
sta = [59,-20,1]
apf = [51,12,10]

# Create numpy array of auditory brain areas in Tailarach coordinates
aud_tal = np.array([a1, a2, sta, apf])

# convert those auditory brain areas from Tailarach to MNI
aud_mni = utils.tal_to_mni(aud_tal)

print aud_mni

"""
Bring up the visualization.
"""
brain = Brain(subject_id, "rh", "inflated")

"""
Now we plot the foci on the inflated surface. We will map
the foci onto the surface by finding the vertex on the "white"
mesh that is closest to the coordinate of each point we want
to display.

While this is not a perfect transformation, it can give you
some idea of where peaks from a volume-based analysis would
be located on the surface.

You can use any valid matplotlib color for the foci; the
default is white.
"""
# plot A1
brain.add_foci(aud_mni[0], map_surface="white", color="gold")

# plot A2
brain.add_foci(aud_mni[1], map_surface="white", color="green")

# plot ST
brain.add_foci(aud_mni[2], map_surface="white", color="blue")

# plot PFC
brain.add_foci(aud_mni[3], map_surface="white", color="red")

# Finally, show everything on screen
mlab.show(stop=True)
