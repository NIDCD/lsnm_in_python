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
#   This file (animate_neural_on_brain.py) was created on December May 13 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 13 2017 
# **************************************************************************/

# animate_neural_on_brain.py
#
# Plays animation of neural activity during one visual DMS trial

#from tvb.simulator.lab import *
#from tvb.simulator.plot.tools import mlab

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# Load one of the cortex 3d surface from TVB data files
#CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")

#plot_surface(CORTEX, op=0.5)

# Load connectivity from Hagmann's brain
#white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
#centres = white_matter.centres

# Plot the 998 nodes of Hagmann's brain, and use the BOLD signal of each node/ROI to
# assign a color and scale each node 
#region_centres = mlab.points3d(centres[:, 0], 
#                               centres[:, 1], 
#                               centres[:, 2],
#                               scale_factor=2.,
#                               color=(0.2,0.4,0.5))
# retrieve current figure and scene's ID

#f0=mlab.gcf()
#scene0 = f0.scene

# starts a mayavi figure for the axial view of the brain
#mlab.figure(f0, bgcolor=(0, 0, 0))

# change the orientation of the brain so we can better observe right hemisphere
#scene0.x_plus_view()

# zoom in to get a closer look of the brain
#scene0.camera.position = [279.42039686733125, -17.694499969482472, 15.823499679565424]
#scene0.camera.focal_point = [0.0, -17.694499969482472, 15.823499679565424]
#scene0.camera.view_angle = 30.0
#scene0.camera.view_up = [0.0, 0.0, 1.0]
#scene0.camera.clipping_range = [140.30522617807688, 455.21171369222844]
#scene0.camera.compute_view_plane_normal()
#scene0.render()

# save figure to a file
#mlab.savefig('brain.png')



# Load data files
lgns = np.loadtxt('lgns.out')
efd1 = np.loadtxt('efd1.out')
efd2 = np.loadtxt('efd2.out')
ev1h = np.loadtxt('ev1h.out')
ev1v = np.loadtxt('ev1v.out')
ev4c = np.loadtxt('ev4c.out')
ev4h = np.loadtxt('ev4h.out')
ev4v = np.loadtxt('ev4v.out')
exfr = np.loadtxt('exfr.out')
exfs = np.loadtxt('exfs.out')
exss = np.loadtxt('exss.out')

# open background image
bg_img = plt.imread('brain_hi_res_cropped.png')

# trial duration in number of timesteps
t_s = 450        # trial starts just after this timestep
t_f = 450        # trial ends just after this timestep
t_d = t_f - t_s  # trial duration 

# Extract number of timesteps from one of the matrices
timesteps = lgns.shape[0]

# Initialize the dimension of each module i.e., dxd = 9x9
d=9

# Reshape all matrices to reflect dimensionality of visual modules
# ... and extract only the first trial (approximately)
lgn = lgns.reshape(timesteps,d,d)
v1h = ev1h.reshape(timesteps,d,d)
v1v = ev1v.reshape(timesteps,d,d)
v4h = ev4h.reshape(timesteps,d,d)
v4c = ev4c.reshape(timesteps,d,d)
v4v = ev4v.reshape(timesteps,d,d)
ss  = exss.reshape(timesteps,d,d)
fd1 = efd1.reshape(timesteps,d,d)
fd2 = efd2.reshape(timesteps,d,d)
fs  = exfs.reshape(timesteps,d,d)
fr  = exfr.reshape(timesteps,d,d)

# plot background image onto figure
fig0 = plt.figure()
ax_0 = fig0.add_axes([0.,0.,1.,1.])
ax_0.imshow(bg_img)
ax_0.axis('off')
ax_1 = fig0.add_axes([.05, .85, .1, .1])
ax_1.imshow(lgn[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_1.axis('off')
ax_2 = fig0.add_axes([.05, .45, .1, .1])
ax_2.imshow(v1h[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_2.axis('off')
ax_3 = fig0.add_axes([.15, .30, .1, .1])
ax_3.imshow(v4h[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_3.axis('off')
ax_4 = fig0.add_axes([.35, .30, .1, .1])
ax_4.imshow(ss[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_4.axis('off')
ax_5 = fig0.add_axes([.65, .45, .1, .1])
ax_5.imshow(fs[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_5.axis('off')
ax_6 = fig0.add_axes([.75, .55, .1, .1])
ax_6.imshow(fd1[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_6.axis('off')
ax_7 = fig0.add_axes([.80, .35, .1, .1])
ax_7.imshow(fd2[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_7.axis('off')
ax_8 = fig0.add_axes([.70, .70, .1, .1])
ax_8.imshow(fr[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
ax_8.axis('off')

# save figure for later use
fig0.savefig('visual_lsnm_on_brain_' + str(t_s) + '.png')

# prepares a grid to display the LSNM arrays distributed across different brain locations
gs = gridspec.GridSpec(4, 9)

# start a second figure
fig1 = plt.figure()

# increase font size
#plt.rcParams.update({'font.size': 30})

#plt.suptitle('SIMULATED NEURAL ACTIVITY')

# Render LGN array in a colormap
ax1 = plt.subplot(gs[0,0])
plt.imshow(lgn[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('LGN')
plt.axis('off')

# Render EV1h array in a colormap
ax2 = plt.subplot(gs[1,1])
plt.imshow(v1h[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('V1h')
plt.axis('off')

# Render EV1v array in a colormap
#plt.subplot(3,4,9)
#plt.imshow(v1v[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('V1v')
#plt.axis('off')

# Render array in a colormap
ax3=plt.subplot(gs[2,2])
plt.imshow(v4h[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('V4h')
plt.axis('off')

# Render array in a colormap
#plt.subplot(3,4,6)
#plt.imshow(v4c[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('V4c')
#plt.axis('off')

# Render array in a colormap
#plt.subplot(3,4,10)
#plt.imshow(v4v[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('V4v')
#plt.axis('off')

# Render array in a colormap
ax4=plt.subplot(gs[3,3])
plt.imshow(ss[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('IT')
plt.axis('off')

# Render array in a colormap
ax5=plt.subplot(gs[1,6])
plt.imshow(fs[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('FS')
plt.axis('off')

# Render array in a colormap
ax6=plt.subplot(gs[0,7])
plt.imshow(fd1[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('FD1')
plt.axis('off')

# Render array in a colormap
ax7=plt.subplot(gs[2,7])
plt.imshow(fd2[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('FD2')
plt.axis('off')

# Render array in a colormap
ax8=plt.subplot(gs[1,8])
plt.imshow(fr[t_s,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
#plt.title('FR')
plt.axis('off')

# increase padding between subplots
#fig.subplots_adjust(hspace=.35)

# Display reference colorbar [left, bottom, width, height]
#cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8])
#cb = plt.colorbar(cax=cbaxes)

# now draw interactive slider at [x, y, length, width]   
axtimesteps = plt.axes([0.1, 0, 0.8, 0.03])

# define current timestep
t_c = t_s

stimesteps = Slider(ax=axtimesteps, label='Timesteps', valmin=t_s, valmax=t_f-1, valinit=t_s)

# now define the function that updates plots as slider is moved
def update(val):
    t_c = stimesteps.val

    plt.subplot(gs[0,0])
    plt.imshow(lgn[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[1,1])
    plt.imshow(v1h[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
#    plt.subplot(3,4,9)
#    plt.imshow(v1v[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[2,2])
    plt.imshow(v4h[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
#    plt.subplot(3,4,6)
#    plt.imshow(v4c[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
#    plt.subplot(3,4,10)
#    plt.imshow(v4v[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[3,3])
    plt.imshow(ss[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[1,6])
    plt.imshow(fs[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[0,7])
    plt.imshow(fd1[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[2,7])
    plt.imshow(fd2[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(gs[1,8])
    plt.imshow(fr[t_c,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
stimesteps.on_changed(update)

#plt.tight_layout()

# Show the plot on the screen
plt.show()
