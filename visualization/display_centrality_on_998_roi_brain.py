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
#   This file (display_centrality_on_998_roi_brain.py) was created on December 14 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on December 14 2017
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# display_centrality_on_998_roi_brain.py
#
# Displays Hagmann's brain's 998-nodes on a transparent brain. The size of each
# node corresponds to either Eigenvector or Betweenness centrality of that node

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm as CM

# load eigenvector centrality into numpy array
pf_ec  = np.load('pf_nodal_ec_w_0.1_density.npy')
pv_ec  = np.load('pv_nodal_ec_w_0.1_density.npy')
dms_ec = np.load('dms_nodal_ec_w_0.1_density.npy')

# load betweenness centrality into numpy array
pf_bc  = np.load('pf_nodal_bc_w_0.1_density.npy')/((998.-1.0)*(998.-2.0))
pv_bc  = np.load('pv_nodal_bc_w_0.1_density.npy')/((998.-1.0)*(998.-2.0))
dms_bc = np.load('dms_nodal_bc_w_0.1_density.npy')/((998.-1.0)*(998.-2.0))

# normalize centrality arrays
min_ec = np.amin(np.concatenate((pf_ec, pv_ec, dms_ec)))
max_ec = np.amax(np.concatenate((pf_ec, pv_ec, dms_ec)))
pf_ec  = (pf_ec  - min_ec) / (max_ec - min_ec)
pv_ec  = (pv_ec  - min_ec) / (max_ec - min_ec)
dms_ec = (dms_ec - min_ec) / (max_ec - min_ec)

# normalize centrality arrays
min_bc = np.amin(np.concatenate((pf_bc, pv_bc, dms_bc)))
max_bc = np.amax(np.concatenate((pf_bc, pv_bc, dms_bc)))
pf_bc  = (pf_bc  - min_bc) / (max_bc - min_bc)
pv_bc  = (pv_bc  - min_bc) / (max_bc - min_bc)
dms_bc = (dms_bc - min_bc) / (max_bc - min_bc)


print 'Minimum centrality was: ', min_ec
print 'Maximum Eigen centrality was: ', max_ec
print 'Minimum Betweenness centrality was: ', min_bc
print 'Maximum Betweenness centrality was: ', max_bc

# Load one of the cortex 3d surface from TVB data files
CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
centres = white_matter.centres

# Starts a new figure to display sagittal view of brain and its nodes
mlab.figure(figure='Sagittal View', bgcolor=(1, 1, 1))

# Plot the 998 nodes of Hagmann's brain, and use the centrality of each node/ROI to
# assign a color and scale each node 
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               #color=(1,0,0))
                               colormap='Reds',
                               scale_factor=5.0)

region_centres.glyph.scale_mode = 'scale_by_vector'
region_centres.mlab_source.dataset.point_data.scalars = dms_bc

# retrieve current figure and scene's ID
f0=mlab.gcf()
scene0 = f0.scene

# change the orientation of the brain so we can better observe right hemisphere
scene0.x_plus_view()

# zoom in to get a closer look of the brain
scene0.camera.position = [279.42039686733125, -17.694499969482472, 15.823499679565424]
scene0.camera.focal_point = [0.0, -17.694499969482472, 15.823499679565424]
scene0.camera.view_angle = 30.0
scene0.camera.view_up = [0.0, 0.0, 1.0]
scene0.camera.clipping_range = [140.30522617807688, 455.21171369222844]
scene0.camera.compute_view_plane_normal()
scene0.render()

# starts another figure for the axial view of the brain
mlab.figure(figure='Axial View', bgcolor=(1, 1, 1))

# Plot the 998 nodes of Hagmann's brain, and use the centrality of each node/ROI to
# assign a color and scale each node 
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               #color=(1,0,0))
                               colormap='Reds',
                               scale_factor=5.0)

region_centres.glyph.scale_mode = 'scale_by_vector'
region_centres.mlab_source.dataset.point_data.scalars = dms_bc

# retrieve current figure and scene's ID
f1=mlab.gcf()
scene1 = f1.scene

# change the orientation of the brain to axial plane
scene1.z_plus_view()

# zoom in to get a closer look of the brain
scene1.camera.position = [1.5493392944335938, -16.746265411376953, 350.48523279890071]
scene1.camera.focal_point = [1.5493392944335938, -16.746265411376953, 19.530120849609375]
scene1.camera.view_angle = 30.0
scene1.camera.view_up = [0.0, 1.0, 0.0]
scene1.camera.clipping_range = [213.86833716646834, 479.14152168211956]
scene1.camera.compute_view_plane_normal()
scene1.render()

# Finally, show everything on screen
mlab.show(stop=True)

# Make a figure to display colorbar.
fig = plt.figure()
ax1 = fig.add_axes([0.05, 0.50, 0.9, 0.05])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.Reds
norm = mpl.colors.Normalize(vmin=min_bc, vmax=max_bc)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
plt.show()
