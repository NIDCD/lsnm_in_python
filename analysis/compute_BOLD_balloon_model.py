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
#   This file (compute_BOLD_balloon_model.py) was created on April 17, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 11 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_fmri_visual.py
#
# Calculate and plot fMRI BOLD signal based using the non-linear balloon model and
# the generalized BOLD signal model (with revised BOLD model coefficients, as
# described in Stephan et al (2007).
# ... using data from visual delay-match-to-sample simulation.
# It also saves the BOLD timeseries for each and all modules in a python data file
# (*.npy)

import numpy as np
import matplotlib.pyplot as plt

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'lsnm_bold_balloon.npy'

# define balloon model parameters...
tau_s = 0.65           # Signal decay in seconds
tau_f = 0.41           # Time of flow-dependent elimination or feedback regulation
                       # in seconds
alpha = 0.32           # Grubb's vessel stiffness exponent
E_0 = 0.4              # Resting oxygen extraction fraction
tau_0 = 0.98           # Hemodynamic transit time in seconds

# define initial conditions for balloon model
s = 1.0                # blood flow
f = 1.0                # blood inflow
v = 1.0                # venous blood volume
q = 1.0                # deoxyhemoglobin content

# define BOLD model parameters...
r_0 = 25.0             # Slope of intravascular relaxation rate (Hz)
nu_0 = 40.3            # Frequency offset at the outer surface of magnetized
                       # vessels (Hz)
epsilon = 0.5          # Ratio of intra- and extravascular signals
V_0 = 4.0              # Resting blood volume fraction
TE = 0.04              # Echo time for a 1.5T scanner

# calculate BOLD model coefficients...
k1 = 4.3 * nu_0 * E_0 * TE
k2 = epsilon * r_0 * E_0 * TE
k3 = 1 - epsilon

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 5 ms.
Ti = 0.005 * 10

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# Time for one complete trial in milliseconds
Ttrial = 5.5

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

# how many scans do you want to remove from beginning of BOLD timeseries?
scans_to_remove = 7

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the document:
#       "Hagmann's Brain Talairach Coordinates (obtained from Barry).doc"
# Provided by Barry Horwitz
# Please note that arrays in Python start from zero so one does to account for that and shift
# indices given by the above document by one location.
# Use all 10 nodes within rPCAL
v1_loc = range(344, 354)

# Use all 22 nodes within rFUS
v4_loc = range(390, 412)

# Use all 6 nodes within rPARH
it_loc = range(412, 418)

# Use all 22 nodes within rRMF
d1_loc =  range(57, 79)

# Use all nodes within rPTRI
d2_loc = range(39, 47)

# Use all nodes within rPOPE
fs_loc = range(47, 57)

# Use all nodes within rCMF
fr_loc = range(125, 138)

# Load TVB nodes synaptic activity
tvb_synaptic = np.load("tvb_synaptic.npy")

# Load TVB host node synaptic activities into separate numpy arrays
tvb_ev1 = tvb_synaptic[:, 0, v1_loc[0]:v1_loc[-1]+1, 0]
tvb_ev4 = tvb_synaptic[:, 0, v4_loc[0]:v4_loc[-1]+1, 0]
tvb_eit = tvb_synaptic[:, 0, it_loc[0]:it_loc[-1]+1, 0]
tvb_ed1 = tvb_synaptic[:, 0, d1_loc[0]:d1_loc[-1]+1, 0]
tvb_ed2 = tvb_synaptic[:, 0, d2_loc[0]:d2_loc[-1]+1, 0]
tvb_efs = tvb_synaptic[:, 0, fs_loc[0]:fs_loc[-1]+1, 0]
tvb_efr = tvb_synaptic[:, 0, fr_loc[0]:fr_loc[-1]+1, 0]
tvb_iv1 = tvb_synaptic[:, 1, v1_loc[0]:v1_loc[-1]+1, 0]
tvb_iv4 = tvb_synaptic[:, 1, v4_loc[0]:v4_loc[-1]+1, 0]
tvb_iit = tvb_synaptic[:, 1, it_loc[0]:it_loc[-1]+1, 0]
tvb_id1 = tvb_synaptic[:, 1, d1_loc[0]:d1_loc[-1]+1, 0]
tvb_id2 = tvb_synaptic[:, 1, d2_loc[0]:d2_loc[-1]+1, 0]
tvb_ifs = tvb_synaptic[:, 1, fs_loc[0]:fs_loc[-1]+1, 0]
tvb_ifr = tvb_synaptic[:, 1, fr_loc[0]:fr_loc[-1]+1, 0]

# Load LSNM synaptic activity data files into a numpy arrays
ev1h = np.loadtxt('ev1h_synaptic.out')
ev1v = np.loadtxt('ev1v_synaptic.out')
iv1h = np.loadtxt('iv1h_synaptic.out')
iv1v = np.loadtxt('iv1v_synaptic.out')
ev4h = np.loadtxt('ev4h_synaptic.out')
ev4v = np.loadtxt('ev4v_synaptic.out')
ev4c = np.loadtxt('ev4c_synaptic.out')
iv4h = np.loadtxt('iv4h_synaptic.out')
iv4v = np.loadtxt('iv4v_synaptic.out')
iv4c = np.loadtxt('iv4c_synaptic.out')
exss = np.loadtxt('exss_synaptic.out')
inss = np.loadtxt('inss_synaptic.out')
efd1 = np.loadtxt('efd1_synaptic.out')
ifd1 = np.loadtxt('ifd1_synaptic.out')
efd2 = np.loadtxt('efd2_synaptic.out')
ifd2 = np.loadtxt('ifd2_synaptic.out')
exfs = np.loadtxt('exfs_synaptic.out')
infs = np.loadtxt('infs_synaptic.out')
exfr = np.loadtxt('exfr_synaptic.out')
infr = np.loadtxt('infr_synaptic.out')

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = ev1h.shape[0]

# Given neural synaptic time interval and total time of scanning experiment,
# construct a numpy array of time points (data points provided in data files)
time_in_seconds = np.arange(0, T, Tr)

# add all units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
v1_syn = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1) + np.sum(tvb_ev1+tvb_iv1, axis=1)
v4_syn = np.sum(ev4h + ev4v + ev4c + iv4h + iv4v + iv4c, axis = 1) + np.sum(tvb_ev4+tvb_iv4, axis=1)
it_syn = np.sum(exss + inss, axis = 1) + np.sum(tvb_eit+tvb_iit, axis=1)
d1_syn = np.sum(efd1 + ifd1, axis = 1) + np.sum(tvb_ed1+tvb_id1, axis=1)
d2_syn = np.sum(efd2 + ifd2, axis = 1) + np.sum(tvb_ed2+tvb_id2, axis=1)
fs_syn = np.sum(exfs + infs, axis = 1) + np.sum(tvb_efs+tvb_ifs, axis=1)
fr_syn = np.sum(exfr + infr, axis = 1) + np.sum(tvb_efr+tvb_ifr, axis=1)

# normalize synaptic activities
max_value = np.amax(v1_syn)
v1_syn = v1_syn / max_value

v1_BOLD = np.zeros(synaptic_timesteps)
for t in range(0, synaptic_timesteps):
    # calculate the change in balloon model variables:
    ds = v1_syn[t] - (1. / tau_s) * s - (1. / tau_f) * (f - 1)
    df = s
    dv = (1. / tau_0) * (f - v ** (1. / alpha))
    dq = (1. / tau_0) * ((f * (1. - (1. - E_0) ** (1. / f)) / E_0) -
                                  (v ** (1. / alpha)) * (q / v))

    # update ballon model variables with rates of change estimated above
    s = s + ds
    f = f + df
    v = v + dv
    q = q + dq

    # now, we need to calculate BOLD signal at each module
    v1_BOLD[t] = V_0 * ((k1 + k2) * (1. - q) + (k3 - k2) * (1. - v))

#v4_BOLD = 
#it_BOLD =
#d1_BOLD =
#d2_BOLD =
#fs_BOLD =
#fr_BOLD =

# now we are going to remove the first trial
# estimate how many 'synaptic ticks' there are in each trial
#synaptic_ticks = Ttrial/Ti
# estimate how many 'MR ticks' there are in each trial
#mr_ticks = round(Ttrial/Tr)

# remove first few scans from BOLD signal array (to eliminate edge effects from
# convolution)
#v1_BOLD = np.delete(v1_BOLD, np.arange(scans_to_remove))
#v4_BOLD = np.delete(v4_BOLD, np.arange(scans_to_remove))
#it_BOLD = np.delete(it_BOLD, np.arange(scans_to_remove))
#d1_BOLD = np.delete(d1_BOLD, np.arange(scans_to_remove))
#d2_BOLD = np.delete(d2_BOLD, np.arange(scans_to_remove))
#fs_BOLD = np.delete(fs_BOLD, np.arange(scans_to_remove))
#fr_BOLD = np.delete(fr_BOLD, np.arange(scans_to_remove))

# create a numpy array of timeseries
#lsnm_BOLD = np.array([v1_BOLD, v4_BOLD, it_BOLD, d1_BOLD, d2_BOLD, fs_BOLD, fr_BOLD])

# now, save all BOLD timeseries to a single file 
#np.save(BOLD_file, lsnm_BOLD)

# Set up figure to plot synaptic activity
plt.figure(1)

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot synaptic activities
plt.plot(v1_syn)
plt.plot(it_syn)
plt.plot(d1_syn)

# Set up separate figures to plot fMRI BOLD signal
plt.figure(2)

plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN V1/V2')

plt.plot(v1_BOLD, linewidth=3.0, color='yellow')
plt.gca().set_axis_bgcolor('black')

print v1_BOLD.shape

#plt.figure(4)

#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN V4')

#plt.plot(v4_BOLD, linewidth=3.0, color='green')
#plt.gca().set_axis_bgcolor('black')

#plt.figure(5)
#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN IT')

#plt.plot(it_BOLD, linewidth=3.0, color='blue')
#plt.gca().set_axis_bgcolor('black')

#plt.figure(6)
#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN D1')

#plt.plot(d1_BOLD, linewidth=3.0, color='red')
#plt.gca().set_axis_bgcolor('black')

# Show the plots on the screen
plt.show()