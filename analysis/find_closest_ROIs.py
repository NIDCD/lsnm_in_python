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
#   This file (find_closest_ROIs.py) was created on 4/29/17,
#   based on 'generate_region_demo_data.py' by Stuart A. Knock and
#            'region_deterministic_bnm_wc.py' by Paula Sanz-Leon,
#            'firing_rate_clamp' by Michael Marmaduke Woodman, and
#            'Evoked Responses in the Visual Cortex', by P. Sanz-Leon.
#
#   This program makes use of The Virtual Brain library toolbox, downloaded
#   from the TVB GitHub page.
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on April 29 2017 
# **************************************************************************/
#
# find_closest_ROIs.py
#
# Finds closest TVB nodes given a set of LSNM module locations
# in Talairach coordinates

from tvb.simulator.lab import connectivity

import scipy.spatial.distance as ds

import numpy as np

# Define connectivity to be used (998 ROI matrix from TVB demo set)
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")

# the following lines of code find the closest Hagmann's brain node to a given
# set of Talairach coordinates
# VISUAL MODEL TALAIRACH COORDINATES
d_m1 = ds.cdist([(-43, -7, 56)], white_matter.centres, 'euclidean')
closest = d_m1[0].argmin()
print 'ROI closest to M1 coordinates given: ', closest, white_matter.centres[closest]


d_v1 = ds.cdist([(18, -88, 8)], white_matter.centres, 'euclidean')
closest = d_v1[0].argmin()
print 'ROI closest to V1 coordinates: ', closest, white_matter.centres[closest]

d_v4 = ds.cdist([(30, -72, -12)], white_matter.centres, 'euclidean')
closest = d_v4[0].argmin()
print 'ROI closest to V4 coordinates: ', closest, white_matter.centres[closest]

d_it = ds.cdist([(28, -36, -8)], white_matter.centres, 'euclidean')
closest = d_it[0].argmin()
print 'ROI closest to IT coordinates: ', closest, white_matter.centres[closest]

d_fs= ds.cdist([(47, 19, 9)], white_matter.centres, 'euclidean')
closest = d_fs[0].argmin()
print 'ROI closest to FS coordinates: ', closest, white_matter.centres[closest]

d_d1= ds.cdist([(42, 26, 20)], white_matter.centres, 'euclidean')
closest = d_d1[0].argmin()
print 'ROI closest to D1 coordinates: ', closest, white_matter.centres[closest]

d_d2= ds.cdist([(38, -37, 36)], white_matter.centres, 'euclidean')
closest = d_d2[0].argmin()
print 'ROI closest to D2 coordinates: ', closest, white_matter.centres[closest]

d_r= ds.cdist([(1, 7, 48)], white_matter.centres, 'euclidean')
closest = d_r[0].argmin()
print 'ROI closest to R coordinates: ', closest, white_matter.centres[closest]

# AUDITORY MODEL TALAIRACH COORDINATES
#d_a1 = ds.cdist([(48, -26, 10)], white_matter.centres, 'euclidean')
#closest = d_a1[0].argsort()[:number_of_closest]
#print closest, white_matter.centres[closest]

#d_a2 = ds.cdist([(62, -32, 10)], white_matter.centres, 'euclidean')
#closest = d_a2[0].argmin()
#closest = d_a2[0].argsort()[:number_of_closest]
#print closest, white_matter.centres[closest]

#d_st = ds.cdist([(59, -17, 4)], white_matter.centres, 'euclidean')
#closest = d_st[0].argsort()[:number_of_closest]
#print closest, white_matter.centres[closest]

#d_pf= ds.cdist([(54, 9, 8)], white_matter.centres, 'euclidean')
#closest = d_pf[0].argsort()[:number_of_closest]
#print closest, white_matter.centres[closest]
