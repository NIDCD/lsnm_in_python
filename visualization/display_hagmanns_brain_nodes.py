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
#   This file (display_hagmanns_brain_nodes.py) was created on June 16, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on June 16, 2015
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# display_hagmanns_brain_nodes.py
#
# Displays Hagmann's brain's 998-nodes plus the hypothetical locations of
# auditory and visual LSNM modules

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

# Load one of the cortex 3d surface from TVB data files
CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
centres = white_matter.centres

# Define the hypothetical Talairach locations of each LSNM visual modules
# Please note that the locations below are the closest locations (to the original
# hypothetical LSNM locations) within Hagmann's brain.
v1 = [14, -86, 7]
v4 = [33,-70,-7]
it = [31,-39,-6]

d1 = [43, 29, 21]
fs = [47, 19, 9]
d2 = [42, 39, 2]
fr = [29, 25, 40]

# The lists of nodes below defines Regions of Interest (ROIs) to be displayed,
# which are the nodes that were used to compute synaptic activity and therefore
# BOLD fMRI activity.
# Use all 10 nodes within rPCAL (LSNM V1 module embedded in TVB node 345)
#v1_loc = range(344, 354)
v1_loc = range(344, 350)

# Use all 22 nodes within rFUS (LSNM V4 module embedded in TVB node 393)
#v4_loc = range(390, 412)
v4_loc = range(390, 396)

# Use all 6 nodes within rPARH (LSNM IT module embedded in TVB node 413)
#it_loc = range(412, 418)
it_loc = range(412, 418)

# Use all 10 nodes within rPOPE (LSNM FS module embedded in TVB node 47)
#fs_loc = range(47, 57)
fs_loc = range(47, 53)

# Use all 22 nodes within rRMF (LSNM D1 module embedded in TVB node 74)
#d1_loc =  range(57, 79)
d1_loc = range(73, 79)

# Use all 8 nodes within rPTRI (LSNM D2 module embedded in TVB node 41)
#d2_loc = range(39, 47)
d2_loc = range(39, 45)

# Use all 13 nodes within rCMF (LSNM FR module embedded in TVB node 125)
#fr_loc = range(125, 138)
fr_loc = range(125, 131)

# calculate the size of each ROI (all have same size so take one)
ROI_size = len(v1_loc)

#fs = [46, 33, 10] ALTERNATE LOCATION
#d2 = [42, 22, 18] ALTERNATE LOCATION
#fr = [38, 19, 32] ALTERNATE LOCATION

# Define the hypothetical Talairach locations of each LSNM auditory modules
a1 = [51,-24,8]
a2 = [61,-36,12]
# Ignore the ST posterior as we are especially interested in ST anterior (for now)
#stp = [60,-39,12]
sta = [59,-20,1]
apf = [51,12,10]

plot_surface(CORTEX, op=0.1)

# Plot the 998 nodes of Hagmann's brain (uncomment if needed for visualization
# purposes
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               color=(0.4, 0.4, 0.4),
                               scale_factor = 2.)

# Now plot the hypothetical locations of LSNM visual modules

# V1 ROI in yellow
#v1_module = mlab.points3d(centres[v1_loc[1],0],
#                          centres[v1_loc[1],1],
#                          centres[v1_loc[1],2],
#                          color=(1, 1, 0),
#                          scale_factor = 6.)

#v1_module = mlab.points3d(centres[v1_loc[0]:v1_loc[ROI_size-1]+1,0],
#                          centres[v1_loc[0]:v1_loc[ROI_size-1]+1,1],
#                          centres[v1_loc[0]:v1_loc[ROI_size-1]+1,2],
#                          color=(1, 1, 0),
#                          scale_factor = 6.)

#print 'Coordinates of the V1 ROI: '
#print centres[v1_loc[0]:v1_loc[ROI_size-1]+1,:]
#print ''

# V4 ROI in green
#v4_module = mlab.points3d(centres[v4_loc[3],0],
#                          centres[v4_loc[3],1],
#                          centres[v4_loc[3],2],
#                          color=(0, 1, 0),
#                          scale_factor = 6.)

#v4_module = mlab.points3d(centres[v4_loc[0]:v4_loc[ROI_size-1]+1,0],
#                          centres[v4_loc[0]:v4_loc[ROI_size-1]+1,1],
#                          centres[v4_loc[0]:v4_loc[ROI_size-1]+1,2],
#                          color=(0, 1, 0),
#                          scale_factor = 6.)

#print 'Coordinates of the V4 ROI: '
#print centres[v4_loc[0]:v4_loc[-1]+1,:]
#print ''

# IT ROI in blue
#it_module = mlab.points3d(centres[it_loc[1],0],
#                          centres[it_loc[1],1],
#                          centres[it_loc[1],2],
#                          color=(0, 0, 1),
#                          scale_factor = 6.)

#it_module = mlab.points3d(centres[it_loc[0]:it_loc[ROI_size-1]+1,0],
#                          centres[it_loc[0]:it_loc[ROI_size-1]+1,1],
#                          centres[it_loc[0]:it_loc[ROI_size-1]+1,2],
#                          color=(0, 0, 1),
#                          scale_factor = 6.)

#print 'Coordinates of the IT ROI: '
#print centres[it_loc[0]:it_loc[-1]+1,:]
#print ''

# FS ROI in orange
#fs_module = mlab.points3d(centres[fs_loc[0],0],
#                          centres[fs_loc[0],1],
#                          centres[fs_loc[0],2],
#                          color=(1, 0.5, 0),
#                          scale_factor = 6.)

#fs_module = mlab.points3d(centres[fs_loc[0]:fs_loc[ROI_size-1]+1,0],
#                          centres[fs_loc[0]:fs_loc[ROI_size-1]+1,1],
#                          centres[fs_loc[0]:fs_loc[ROI_size-1]+1,2],
#                          color=(1, 0.5, 0),
#                          scale_factor = 6.)

#print 'Coordinates of the FS ROI: '
#print centres[fs_loc[0]:fs_loc[-1]+1,:]
#print ''

# D1 ROI in red
#d1_module = mlab.points3d(centres[d1_loc[1],0],
#                          centres[d1_loc[1],1],
#                          centres[d1_loc[1],2],
#                          color=(1, 0, 0),
#                          scale_factor = 6.)

#d1_module = mlab.points3d(centres[d1_loc[0]:d1_loc[ROI_size-1]+1,0],
#                          centres[d1_loc[0]:d1_loc[ROI_size-1]+1,1],
#                          centres[d1_loc[0]:d1_loc[ROI_size-1]+1,2],
#                          color=(1, 0, 0),
#                          scale_factor = 6.)

#print 'Coordinates of the D1 ROI: '
#print centres[d1_loc[0]:d1_loc[-1]+1,:]
#print ''

# D2 ROI in magenta
#d2_module = mlab.points3d(centres[d2_loc[2],0],
#                          centres[d2_loc[2],1],
#                          centres[d2_loc[2],2],
#                          color=(1, 0, 1),
#                          scale_factor = 6.)

#d2_module = mlab.points3d(centres[d2_loc[0]:d2_loc[ROI_size-1]+1,0],
#                          centres[d2_loc[0]:d2_loc[ROI_size-1]+1,1],
#                          centres[d2_loc[0]:d2_loc[ROI_size-1]+1,2],
#                          color=(1, 0, 1),
#                          scale_factor = 6.)

#print 'Coordinates of the D2 ROI: '
#print centres[d2_loc[0]:d2_loc[-1]+1,:]
#print ''

# FR ROI in purple
#fr_module = mlab.points3d(centres[fr_loc[0],0],
#                          centres[fr_loc[0],1],
#                          centres[fr_loc[0],2],
#                          color=(0.5, 0, 0.5),
#                          scale_factor = 6.)

#fr_module = mlab.points3d(centres[fr_loc[0]:fr_loc[ROI_size-1]+1,0],
#                          centres[fr_loc[0]:fr_loc[ROI_size-1]+1,1],
#                          centres[fr_loc[0]:fr_loc[ROI_size-1]+1,2],
#                          color=(0.5, 0, 0.5),
#                          scale_factor = 6.)

#print 'Coordinates of the FR ROI: '
#print centres[fr_loc[0]:fr_loc[-1]+1,:]
#print ''

# Now plot the hypothetical locations of LSNM auditory modules
a1_module = mlab.points3d(a1[0],a1[1],a1[2],color=(1, 1, 0),scale_factor = 10.)
a2_module = mlab.points3d(a2[0],a2[1],a2[2],color=(0, 1, 0),scale_factor = 10.)
sta_module = mlab.points3d(sta[0],sta[1],sta[2],color=(0, 0, 1),scale_factor = 10.)
apf_module = mlab.points3d(apf[0],apf[1],apf[2],color=(1, 0, 0),scale_factor = 10.)

# Finally, show everything on screen
mlab.show(stop=True)
