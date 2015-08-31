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
#   This file (compute_BOLD_poisson_model.py) was created on April 17, 2015.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on August 11 2015
#
#   Based on computer code originally developed by Barry Horwitz et al
# **************************************************************************/

# compute_BOLD_poisson_model.py
#
# Calculate and plot fMRI BOLD signal using a convolution with hemodynamic response
# function modeled as a poisson time-series, based on data from visual
# delay-match-to-sample simulation. It also saves the BOLD timeseries for each
# and all modules in a python data file (*.npy)


import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import poisson

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'lsnm_bold.npy'

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 5 ms.
Ti = 0.005 * 10

# define constant needed for hemodynamic function (in milliseconds)
lambda_ = 6

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

# what are the locations of relevant TVB nodes within TVB array?
#v1_loc = 345
#v4_loc = 393
#it_loc = 413
#pf_loc =  74
# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the document:
#       "Hagmann's Brain Talairach Coordinates (obtained from Barry).doc"
# Provided by Barry Horwitz
# Please note that arrays in Python start from zero so one does have to account for
# that and shift indices given by the above document by one location.
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

# the following calculates a Poisson distribution (that will represent a hemodynamic
# function, given lambda (the Poisson time constant characterizing width and height
# of hemodynamic function), and tau (the time step)
# if you would do it manually you would do the following:
#h = [lambda_ ** tau * m.exp(-lambda_) / m.factorial(tau) for tau in time_in_seconds]
h = poisson.pmf(time_in_seconds, lambda_)

# the following calculates the impulse response (convolution kernel) of the gamma
# function, approximating the BOLD response (Boynton model).

# parameters for HRF gamma function
# time constant
#tau = 1.25
# phase delay
#n = 3
# pure delay
#delta = 2.5
# this is just one of the calculations needed for the gamma function
#a = (time_in_seconds - delta) / tau
# 'clip' out negative numbers, if any
#a = a.clip(min=0)
# calculate the array containing the time points of the gamma function
#h = [(x ** (n-1) * m.exp(-x)) / (tau * m.factorial(n-1)) for x in a]
# now, convert to a numpy array
#h = np.asarray(h)

# rescale the array containing the poisson to increase its size and match the size of
# the synaptic activity array (using linear interpolation)
scanning_timescale = np.arange(0, synaptic_timesteps, synaptic_timesteps / (T/Tr))
synaptic_timescale = np.arange(0, synaptic_timesteps)
h = np.interp(synaptic_timescale, scanning_timescale, h)

# add all units WITHIN each region together across space to calculate
# synaptic activity in EACH brain region
v1_syn = np.sum(ev1h + ev1v + iv1h + iv1v, axis = 1) + np.sum(tvb_ev1+tvb_iv1, axis=1)
v4_syn = np.sum(ev4h + ev4v + ev4c + iv4h + iv4v + iv4c, axis = 1) + np.sum(tvb_ev4+tvb_iv4, axis=1)
it_syn = np.sum(exss + inss, axis = 1) + np.sum(tvb_eit+tvb_iit, axis=1)
d1_syn = np.sum(efd1 + ifd1, axis = 1) + np.sum(tvb_ed1+tvb_id1, axis=1)
d2_syn = np.sum(efd2 + ifd2, axis = 1) + np.sum(tvb_ed2+tvb_id2, axis=1)
fs_syn = np.sum(exfs + infs, axis = 1) + np.sum(tvb_efs+tvb_ifs, axis=1)
fr_syn = np.sum(exfr + infr, axis = 1) + np.sum(tvb_efr+tvb_ifr, axis=1)

# now, we need to convolve the synaptic activity with a hemodynamic delay
# function and sample the array at Tr regular intervals (back to the scanning
# timescale)
v1_BOLD = np.convolve(v1_syn, h)[scanning_timescale]
v4_BOLD = np.convolve(v4_syn, h)[scanning_timescale]
it_BOLD = np.convolve(it_syn, h)[scanning_timescale]
d1_BOLD = np.convolve(d1_syn, h)[scanning_timescale]
d2_BOLD = np.convolve(d2_syn, h)[scanning_timescale]
fs_BOLD = np.convolve(fs_syn, h)[scanning_timescale]
fr_BOLD = np.convolve(fr_syn, h)[scanning_timescale]

# now we are going to remove the first trial
# estimate how many 'synaptic ticks' there are in each trial
synaptic_ticks = Ttrial/Ti
# estimate how many 'MR ticks' there are in each trial
mr_ticks = round(Ttrial/Tr)

# remove first few scans from BOLD signal array (to eliminate edge effects from
# convolution)
v1_BOLD = np.delete(v1_BOLD, np.arange(scans_to_remove))
v4_BOLD = np.delete(v4_BOLD, np.arange(scans_to_remove))
it_BOLD = np.delete(it_BOLD, np.arange(scans_to_remove))
d1_BOLD = np.delete(d1_BOLD, np.arange(scans_to_remove))
d2_BOLD = np.delete(d2_BOLD, np.arange(scans_to_remove))
fs_BOLD = np.delete(fs_BOLD, np.arange(scans_to_remove))
fr_BOLD = np.delete(fr_BOLD, np.arange(scans_to_remove))

# create a dictionary of timeseries indexed by module name
lsnm_BOLD = np.array([v1_BOLD, v4_BOLD, it_BOLD, d1_BOLD, d2_BOLD, fs_BOLD, fr_BOLD])

print lsnm_BOLD.shape

# now, save all BOLD timeseries to a single file 
np.save(BOLD_file, lsnm_BOLD)

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

plt.figure(4)

plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN V4')

plt.plot(v4_BOLD, linewidth=3.0, color='green')
plt.gca().set_axis_bgcolor('black')

plt.figure(5)
plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN IT')

plt.plot(it_BOLD, linewidth=3.0, color='blue')
plt.gca().set_axis_bgcolor('black')

plt.figure(6)
plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN D1')

plt.plot(d1_BOLD, linewidth=3.0, color='red')
plt.gca().set_axis_bgcolor('black')

# Show the plots on the screen
plt.show()
