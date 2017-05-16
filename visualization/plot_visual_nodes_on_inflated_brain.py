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
#   This file (plot_visual_nodes_on_inflated_brain.py) was created on November 3 2016.
#
#
#   Author: Antonio Ulloa. Based on Pysurfer's plot_foci.py
#
#   Last updated by Antonio Ulloa on November 3 2016
#
# **************************************************************************/
#
# plot_visual_nodes_on_inflated_brain.py
#
# Displays right hemisphere of PySurfer's inflated brain and superimposes spheres
# representing visual short-term memory model brain areas.

import os
import nibabel as nib
from surfer import Brain
from surfer import utils

from mayavi import mlab

import numpy as np

print(__doc__)

subject_id = "fsaverage"
subjects_dir = os.environ["SUBJECTS_DIR"]

# Define the hypothetical Talairach locations of each LSNM visual modules
# Please note that the locations below are the closest locations (to the original
# hypothetical LSNM locations) within Hagmann's brain.
v1 = [14, -86, 7]
v4 = [33,-70,-7]
it = [31,-39,-6]

fs = [47, 19, 9]
d1 = [43, 29, 21]
d2 = [42, 39, 2]
fr = [29, 25, 40]

# Create numpy array of visual brain areas in Tailarach coordinates
vis_tal = np.array([v1, v4, it, fs, d1, d2, fr])

# convert those visual brain areas from Tailarach to MNI
vis_mni = utils.tal_to_mni(vis_tal)

print vis_mni

"""
Bring up the visualization.
"""
brain = Brain(subject_id, "rh", "inflated", cortex='high_contrast')

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
brain.add_foci(vis_mni[0], map_surface="white", color="gold")

# plot A2
brain.add_foci(vis_mni[1], map_surface="white", color="green")

# plot IT
brain.add_foci(vis_mni[2], map_surface="white", color="blue")

# plot FS
brain.add_foci(vis_mni[3], map_surface="white", color="orange")

# plot D1
brain.add_foci(vis_mni[4], map_surface="white", color="red")

# plot D2
brain.add_foci(vis_mni[5], map_surface="white", color="pink")

# plot FR
brain.add_foci(vis_mni[6], map_surface="white", color="purple")


# Finally, show everything on screen
mlab.show(stop=True)
