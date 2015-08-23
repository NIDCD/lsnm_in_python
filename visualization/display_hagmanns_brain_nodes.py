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
v1 = [14, -86, 7]
v4 = [33,-70,-7]
it = [31,-39,-6]

d1 = [43, 29, 21]
fs = [47, 19, 9]
d2 = [42, 39, 2]
fr = [29, 25, 40]

#fs = [46, 33, 10] ALTERNATE LOCATION
#d2 = [42, 22, 18] ALTERNATE LOCATION
#fr = [38, 19, 32] ALTERNATE LOCATION

# Define the hypothetical Talairach locations of each LSNM auditory modules
a1 = [48,-26,10]
a2 = [62,-32,10]
# Ignore the posterior ST as we are especially interested in anterior (for now)
#stp = [60,-39,12]
sta = [59,-17,4]
apf = [56,21,5]

plot_surface(CORTEX, op=0.3)

# Plot the 998 nodes of Hagmann's brain
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               color=(0.4, 0.4, 0.4),
                               scale_factor = 2.)

# Now plot the hypothetical locations of LSNM visual modules
# v1 is yellow, v4 is green, it is blue, pf is red
# V1 in yellow
v1_module = mlab.points3d(v1[0],v1[1],v1[2],color=(1, 1, 0),scale_factor = 10.)
# V4 in green
v4_module = mlab.points3d(v4[0],v4[1],v4[2],color=(0, 1, 0),scale_factor = 10.)
# IT in blue
it_module = mlab.points3d(it[0],it[1],it[2],color=(0, 0, 1),scale_factor = 10.)
# FS in orange
fs_module = mlab.points3d(fs[0],fs[1],fs[2],color=(1, 0.5, 0),scale_factor = 10.)
# D1 in red
d1_module = mlab.points3d(d1[0],d1[1],d1[2],color=(1, 0, 0),scale_factor = 10.)
# D2 in magenta
d2_module = mlab.points3d(d2[0],d2[1],d2[2],color=(1, 0, 1),scale_factor = 10.)
# FR in purple
fr_module = mlab.points3d(fr[0],fr[1],fr[2],color=(0.5, 0, 0.5),scale_factor = 10.)

# Now plot the hypothetical locations of LSNM auditory modules
#a1_module = mlab.points3d(a1[0],a1[1],a1[2],color=(1, 1, 0),scale_factor = 10.)
#a2_module = mlab.points3d(a2[0],a2[1],a2[2],color=(0, 1, 0),scale_factor = 10.)
####IGNORE STP FOR NOW
#stp_module = mlab.points3d(stp[0],stp[1],stp[2],color=(0, 1, 1),scale_factor = 10.)
####IGNORE STP FOR NOW
#sta_module = mlab.points3d(sta[0],sta[1],sta[2],color=(0, 0, 1),scale_factor = 10.)
#apf_module = mlab.points3d(apf[0],apf[1],apf[2],color=(1, 0, 0),scale_factor = 10.)

# Finally, show everything on screen
mlab.show(stop=True)
