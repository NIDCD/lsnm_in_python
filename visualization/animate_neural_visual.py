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
#   This file (animate_neural_visual.py) was created on December January 22, 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on September 12 2017  
# **************************************************************************/

# animate_neural_visual.py
#
# Plays a movie using output data files of visual delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# Extract number of timesteps from one of the matrices
timesteps = lgns.shape[0]

# Initialize the dimension of each module i.e., dxd = 9x9
d=9

# Reshape all matrices to reflect dimensionality of visual modules
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

fig, ax = plt.subplots()

# increase font size
plt.rcParams.update({'font.size': 30})

#plt.suptitle('SIMULATED NEURAL ACTIVITY')

# Render LGN array in a colormap
plt.subplot(3,4,1)
plt.imshow(lgn[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('LGN')
plt.axis('off')

# Render EV1h array in a colormap
plt.subplot(3,4,5)
plt.imshow(v1h[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('V1h')
plt.axis('off')

# Render EV1v array in a colormap
plt.subplot(3,4,9)
plt.imshow(v1v[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('V1v')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,2)
plt.imshow(v4h[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('V4h')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,6)
plt.imshow(v4c[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('V4c')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,10)
plt.imshow(v4v[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('V4v')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,3)
plt.imshow(ss[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('IT')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,7)
plt.imshow(fs[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('FS')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,11)
plt.imshow(fd1[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('FD1')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,4)
plt.imshow(fd2[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('FD2')
plt.axis('off')

# Render array in a colormap
plt.subplot(3,4,8)
plt.imshow(fr[0,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
plt.title('FR')
plt.axis('off')

# increase padding between subplots
fig.subplots_adjust(hspace=.35)

# Display reference colorbar [left, bottom, width, height]
#cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8])
#cb = plt.colorbar(cax=cbaxes)

# now draw interactive slider at [x, y, length, width]   
axtimesteps = plt.axes([0.1, 0, 0.8, 0.03])

stimesteps = Slider(axtimesteps, 'Slider', 0, timesteps-1, valinit=0)

# now define the function that updates plots as slider is moved
def update(val):
    timesteps = stimesteps.val

    plt.subplot(3,4,1)
    plt.imshow(lgn[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,5)
    plt.imshow(v1h[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,9)
    plt.imshow(v1v[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,2)
    plt.imshow(v4h[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,6)
    plt.imshow(v4c[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,10)
    plt.imshow(v4v[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,3)
    plt.imshow(ss[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,7)
    plt.imshow(fs[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,11)
    plt.imshow(fd1[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,4)
    plt.imshow(fd2[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
    plt.subplot(3,4,8)
    plt.imshow(fr[timesteps,:,:], vmin=0, vmax=1, cmap='hot', interpolation='none')
    
stimesteps.on_changed(update)

#plt.tight_layout()

# Show the plot on the screen
plt.show()
