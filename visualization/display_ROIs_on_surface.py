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
#   This file (display_ROIs_on_surface.py) was created on August 31, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on June 16, 2015
#
#   Based on: display_surface_parcellation.py by Stuart Knock (TVB team)
# **************************************************************************/
#
# display_ROIs_on_surface.py
#
# Displays a surface of the cerebral cortex that shows the location of
# the regions of interest (ROIs) under study, labeled by given colors.

from tvb.simulator.lab import *
from tvb.simulator.region_boundaries import RegionBoundaries
from tvb.simulator.region_colours import RegionColours 


CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")
CORTEX_BOUNDARIES = RegionBoundaries(CORTEX)

#region_colours = RegionColours(CORTEX_BOUNDARIES.region_neighbours)
#colouring = region_colours.back_track()

number_of_regions = len(CORTEX_BOUNDARIES.region_neighbours)

#for k in range(int(number_of_regions)):
#    colouring[k + int(number_of_regions)] = colouring[k]


mapping_colours = list("rgbcmyRGBCMY")
colour_rgb = {"r": numpy.array([255,   0,   0], dtype=numpy.uint8),
              "g": numpy.array([  0, 255,   0], dtype=numpy.uint8),
              "b": numpy.array([  0,   0, 255], dtype=numpy.uint8),
              "c": numpy.array([  0, 255, 255], dtype=numpy.uint8),
              "m": numpy.array([255,   0, 255], dtype=numpy.uint8),
              "y": numpy.array([255, 255,   0], dtype=numpy.uint8),
              "R": numpy.array([128,   0,   0], dtype=numpy.uint8),
              "G": numpy.array([  0, 128,   0], dtype=numpy.uint8),
              "B": numpy.array([  0,   0, 128], dtype=numpy.uint8),
              "C": numpy.array([  0, 128, 128], dtype=numpy.uint8),
              "M": numpy.array([128,   0, 128], dtype=numpy.uint8),
              "Y": numpy.array([128, 128,   0], dtype=numpy.uint8)}


#(surf_mesh, bpts) = surface_parcellation(CORTEX_BOUNDARIES, colouring, mapping_colours, colour_rgb, interaction=True)



##- EoF -##
