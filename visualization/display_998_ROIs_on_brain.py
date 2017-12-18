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
#   This file (display_998_ROIs_on_brain.py) was created on May 13 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 9 2017
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# animate_BOLD_using_Hagmann_nodes.py
#
# Displays Hagmann's brain's 998-nodes on a transparent brain. The color of each
# node corresponds to fMRI BOLD activity at a given point in time

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm as CM

# Load one of the cortex 3d surface from TVB data files
CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
centres = white_matter.centres

# Load BOLD fMRI timeseries
#ts = np.load('tvb_neuronal.npy')

# reshape the obtained array to extract only the timeseries 
#ts = ts[:,0,:,0].T

current_timepoint = 425

# create a colormap that gives hot colors to higher BOLD values and
# cool colors to lower BOLD values, normalized to the higher value over the
# whole timeseries contained synaptic input file (duration of a full DMS trial)
#max_ts = np.amax(ts)
#colors=ts[:, current_timepoint]/max_ts



#print 'Dimensions of timeseries array: ', ts.shape

#plot_surface(CORTEX, op=0.05)

# Starts a new figure to display sagittal view of brain and its nodes
mlab.figure(figure='Sagittal View', bgcolor=(0, 0, 0))


# Plot the 998 nodes of Hagmann's brain, and use the BOLD signal of each node/ROI to
# assign a color and scale each node 
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               scale_factor=2.)


#rc = region_centres.mlab_source
#scalars=colors
#rc.set(scalars=scalars)
region_centres.glyph.scale_mode = 'scale_by_vector'
#region_centres.mlab_source.dataset.point_data.scalars = colors

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
mlab.figure(figure='Axial View', bgcolor=(0, 0, 0))

# Plot the 998 nodes of Hagmann's brain, and use the BOLD signal of each node/ROI to
# assign a color and scale each node 
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               scale_factor=2.)
#rc = region_centres.mlab_source
#scalars=colors
#rc.set(scalars=scalars)
region_centres.glyph.scale_mode = 'scale_by_vector'
#region_centres.mlab_source.dataset.point_data.scalars = colors

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

# now we are going to animate the nodes on the brain to change the color of the nodes
# as we progress through the simulation (stored in a npy file)
#@mlab.animate(delay=10)
#def anim():
#    rc = region_centres.mlab_source
#    for current_ts in range(ts.shape[1]):
#        print 'Updating scene... ts # ', current_ts
#        colors = ts[:, current_ts]/max_ts
#        scalars = colors
#        rc.set(scalars=scalars)
#        yield
#
#anim()
            
# Finally, show everything on screen
mlab.show(stop=True)

#create figure with subplots
#f, (ax1, ax2, ax3) = plt.subplots(1, 3)
f = plt.figure()

# plot input presented to the simulated subject ('O' shape)
#input=np.zeros([9,9])
#input[2,2]=1
#input[2,3]=1
#input[2,4]=1
#input[2,5]=1
#input[2,6]=1
#input[6,2]=1
#input[6,3]=1
#input[6,4]=1
#input[6,5]=1
#input[6,6]=1
#input[3,2]=1
#input[4,2]=1
#input[5,2]=1
#input[3,6]=1
#input[4,6]=1
#input[5,6]=1
#ITI=np.zeros([9,9])
#ax1.imshow(input,interpolation='nearest', cmap='Greys')
#ax1.annotate('intertrial interval', xy=(1,4))
#ax1.annotate('delay period', xy=(1.5,4))
#ax1.axis('off')

# plot 998 ROIs on brain sagittal view in subplot
brain_array_0 = mlab.screenshot(figure=f0)
plt.imshow(brain_array_0)
plt.axis('off')

# plot 998 ROIs on brain axial view in second subplot
#brain_array_1 = mlab.screenshot(figure=f1)
#ax3.imshow(brain_array_1)
#ax3.axis('off')

# save figure for later use
f.savefig('brain_nodes_screenshot_' + str(current_timepoint) +'.png')

plt.show()
