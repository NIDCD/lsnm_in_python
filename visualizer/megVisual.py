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
#   This file (megVisual.py) was created on June 7, 2015.
#
#   Based on Sanz-Leon et al (2015) and Sarvas (1987) and on TVB's monitors.py
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on June 7 2015  
# **************************************************************************/

# megVisual.py
#
# Calculate and plot MEG signal at source locations based on data from visual
# delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

#the magnetic constant = 1.25663706 × 10-6 m kg s-2 A-2  (H/m)
mu_0 = 1.25663706e-6 #mH/mm

# hypothesized Talairach coordinates of LSNM brain regions
v1_loc = [18, -88, 8]

v4_loc = [30, -72, -12]

it_loc = [28, -36, -8]

pf_loc = [42, 26, 20]

# initialize source positions
r_0 = [v1_loc, v4_loc, it_loc, pf_loc] 

#initialize vector from sources to sensor
Q = simulator.connectivity.orientations

centre = numpy.mean(r_0, axis=0)[numpy.newaxis, :]
radius = 1.01 * max(numpy.sqrt(numpy.sum((r_0 - centre)**2, axis=1)))

# Load V1 synaptic activity data files into a numpy array
ev1h = np.loadtxt('../simulator/output/ev1h_synaptic.out')
ev1v = np.loadtxt('../simulator/output/ev1v_synaptic.out')

# Load V4 synaptic activity data files into a numpy array
ev4h = np.loadtxt('../simulator/output/ev4h_synaptic.out')
ev4c = np.loadtxt('../simulator/output/ev4c_synaptic.out')
ev4v = np.loadtxt('../simulator/output/ev4v_synaptic.out')

# Load IT synaptic activity data files into a numpy array
exss = np.loadtxt('../simulator/output/exss_synaptic.out')

# Load PFC synaptic activity data files into a numpy array
efd1 = np.loadtxt('../simulator/output/efd1_synaptic.out')
efd2 = np.loadtxt('../simulator/output/efd2_synaptic.out')
exfs = np.loadtxt('../simulator/output/exfs_synaptic.out')
exfr = np.loadtxt('../simulator/output/exfr_synaptic.out')

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = ev1h.shape[0]

# add all units within each region (V1, V4, IT, D1, D2, FS, R) together across space to calculate
# MEG source dynamics in each brain region
v1 = np.sum(ev1h + ev1v, axis = 1)
v4 = np.sum(ev4h + ev4c + ev4v, axis=1)
it = np.sum(exss, axis = 1)
d1 = np.sum(efd1, axis = 1)
d2 = np.sum(efd2, axis = 1)
fs = np.sum(exfs, axis = 1)
fr = np.sum(exfr, axis = 1)

# Set up figure to plot MEG source dynamics
plt.figure(1)

plt.suptitle('SIMULATED MEG SOURCE DYNAMICS')

# Plot MEG signal
v1_plot=plt.plot(v1, label='V1')
v4_plot=plt.plot(v4, label='V4')
it_plot=plt.plot(it, label='IT')
d1_plot=plt.plot(d1, label='D1')
d2_plot=plt.plot(d2, label='D2')
fs_plot=plt.plot(fs, label='FS')
fr_plot=plt.plot(fr, label='FR')

plt.legend()

# Show the plot on the screen
plt.show()
