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
#   This file (compute_BOLD_balloon_998_regions.py) was created on March 8 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on March 8 2017
#
# **************************************************************************/

# compute_BOLD_balloon_998_regions.py
#
# Calculate and plot fMRI BOLD signal, using the
# Balloon model, as described by Stephan et al (2007)
# and Friston et al (2000), and
# the BOLD signal model, as described by Stephan et al (2007) and
# Friston et al (2000).
# Parameters for both were taken from Friston et al (2000) and they were
# estimated using a 2T scanner, with a TR of 2 seconds, 
# 
# ... using data from visual delay-match-to-sample simulation, 998 ROIs from
# Hagmann's connectome.
# It also saves the BOLD timeseries for each and all ROIs in a python data file
# (*.npy)
#
# Finally, the functional connectivity matrix is displayed and saved in a python data file.
# ... and we also display a histogram of frequencies of correlations. And we calculate
# and print Fisher's kurtosis and skewness of cross-correlation coefficients.
#

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint

from scipy.stats import kurtosis

from scipy.stats import skew

from matplotlib import cm as CM

# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_998_ROIs.npy'

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'bold_balloon_998_regions.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
xcorr_file = 'xcorr_matrix_998_regions.npy'

# define balloon model parameters...
tau_s = 1.5           # rate constant of vasodilatory signal decay in seconds
                      # from Friston et al, 2000

tau_f = 4.5           # Time of flow-dependent elimination or feedback regulation
                      # in seconds, from Friston et al, 2000

alpha = 0.2           # Grubb's vessel stiffness exponent, from Friston et al, 2000

tau_0 = 1.0           # Hemodynamic transit time in seconds
                      # from Friston et al, 2000

epsilon = 0.1         # efficacy of synaptic activity to induce the signal,
                      # from Friston et al, 2000
                      
# define BOLD model parameters...
r_0 = 25.0            # Slope of intravascular relaxation rate (Hz)
                      # (Obata et al, 2004)

nu_0 = 40.3           # Frequency offset at the outer surface of magnetized
                      # vessels (Hz) (Obata et al, 2004)

e = 1.43        # Ratio of intra- and extravascular BOLD signal at rest
                       # (Obata et al, 2004)

V_0 = 0.02            # Resting blood volume fraction, from Friston et al, 2000

E_0 = 0.8             # Resting oxygen extraction fraction (Friston et al, 2000)

TE = 0.040            # Echo time for a 1.5T scanner (Friston et al, 2000)

# calculate ratio of intra- and extravascular BOLD signal at rest
#mu_epsilon = 1.0
#nu_epsilon = 
#epsilon = mu_epsilon * exp(nu_epsilon)

# calculate BOLD model coefficients, from Friston et al (2000)
#k1 = 7.0 * E_0
#k2 = 2.0
#k3 = 2.0 * E_0 - 0.2 
k1 = 4.3 * nu_0 * E_0 * TE
k2 = e * r_0 * E_0 * TE
k3 = 1.0 - epsilon

def balloon_function(y, t, syn):
    ''' 
    Balloon model of hemodynamic change
    
    '''
    
    # unpack initial values
    s = y[0]
    f = y[1]
    v = y[2]
    q = y[3]

    x = syn[np.floor(t * synaptic_timesteps / T)]

    # the balloon model equations
    ds = epsilon * x - (1. / tau_s) * s - (1. / tau_f) * (f - 1)
    df = s
    dv = (1. / tau_0) * (f - v ** (1. / alpha))
    dq = (1. / tau_0) * ((f * (1. - (1. - E_0) ** (1. / f)) / E_0) -
                          (v ** (1. / alpha)) * (q / v))

    return [ds, df, dv, dq]

# define neural synaptic time interval in seconds. The simulation data is collected
# one data point at synaptic intervals (10 simulation timesteps). Every simulation
# timestep is equivalent to 5 ms.
Ti = 0.005 * 10

# Total time of scanning experiment in seconds (timesteps X 5)
T = 198

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

# how many scans do you want to remove from beginning of BOLD timeseries?
scans_to_remove = 8

# read the input file that contains the synaptic activities of all ROIs
syn = np.load(SYN_file)

# claculate total number of ROIs
ROIs = syn.shape[0]

print 'Dimensions of synaptic file: ', syn.shape
print 'LENGTH OF SYNAPTIC TIME-SERIES: ', syn[0].size
print 'Number of ROIs: ', ROIs

# Throw away first value of each synaptic array (it is always zero)
syn = syn[:, 1:]

# normalize synaptic activities (needed prior to BOLD estimation)
for idx, roi in enumerate(syn):
    syn[idx] = (syn[idx] - syn[idx].min()) / (syn[idx].max() - syn[idx].min())

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = syn[0].size
print 'Size of synaptic arrays: ', synaptic_timesteps

# Given neural synaptic time interval and total time of scanning experiment,
# construct a numpy array of time points (data points provided in data files)
# Note that we subtract 1 from simulation time bc we threw away first timepoint for
# the integrated synaptic activity
time_in_seconds = np.arange(0, T, Ti)
time_in_seconds = time_in_seconds[:-1]

# Hard coded initial conditions
s_0 = 0   # s, blood flow
f_0 = 1.  # f, blood inflow
v_0 = 1.  # v, venous blood volume
q_0 = 1.  # q, deoxyhemoglobin content

# initial conditions vector (one for each of the 998 ROIs)
y_0 = np.zeros([ROIs, 4])
for idx in range(0, ROIs):
    y_0[idx] = [s_0, f_0, v_0, q_0]

print 'Initial conditions vector OK, is of shape: ', y_0.shape
    
# generate synaptic time array
t_syn = np.arange(0, synaptic_timesteps)

# generate time array for solution
t = time_in_seconds

print 'Size of real time vector is: ', t.size

# solve the ODEs for given initial conditions, parameters, and timesteps
state = np.zeros([t.size, ROIs, 4])
print 'Solving ODEs...'
for idx in range(0, ROIs):
    state[:, idx] = odeint(balloon_function, y_0[idx], t, args=(syn[idx],) )


# Unpack the state variables used in the BOLD model
s = np.zeros([ROIs, t.size])
f = np.zeros([ROIs, t.size])
v = np.zeros([ROIs, t.size])
q = np.zeros([ROIs, t.size])

for idx in range(0, ROIs):
    s[idx] = state[:, idx, 0]
    f[idx] = state[:, idx, 1]
    v[idx] = state[:, idx, 2]
    q[idx] = state[:, idx, 3]

# now, we need to calculate BOLD signal at each timestep, based on v and q obtained from solving
# balloon model ODE above.
BOLD = np.zeros([ROIs, t.size])
for idx in range(0, ROIs):
    BOLD[idx] = np.array(V_0 * (k1 * (1. - q[idx]) + k2 * (1. - q[idx] / v[idx]) + k3 * (1. - v[idx])) )

# The following is for display purposes only:
# Construct a numpy array of timesteps (data points provided in data file)
# to convert from timesteps to time in seconds we do the following:
# Each simulation time-step equals 5 milliseconds
# However, we are recording only once every 10 time-steps
# Therefore, each data point in the output files represents 50 milliseconds.
# Thus, we need to multiply the datapoint times 50 ms...
# ... and divide by 1000 to convert to seconds
#t = np.linspace(0, 659*50./1000., num=660)
t = np.linspace(0, synaptic_timesteps+1 * 50.0 / 1000., num=synaptic_timesteps+1)

# downsample the BOLD signal arrays to produce scan rate of 2 per second
scanning_timescale = np.linspace(0, synaptic_timesteps-1, num=T/Tr)
scanning_timescale = np.round(scanning_timescale)
scanning_timescale = scanning_timescale.astype(int)
mr_time = t[scanning_timescale]

down_sampled_BOLD = np.zeros([ROIs, mr_time.size])

print 'Scanning timescale: ', scanning_timescale

print 'Size of scanning timescale array:', scanning_timescale.size

print 'Size of BOLD array before downsampling: ', BOLD.shape

for idx in range(0, ROIs):
    down_sampled_BOLD[idx] = BOLD[idx, scanning_timescale]

print 'Size of BOLD arrays after downsampling: ', down_sampled_BOLD.shape

# remove first few scans from BOLD signal array and from BOLD timescale array
mr_time = np.delete(mr_time, np.arange(scans_to_remove))
fMRI_BOLD = np.delete(down_sampled_BOLD, np.arange(scans_to_remove), 1)

# round of mr time for display purposes
mr_time = np.round(mr_time, decimals=0)

print 'Size of BOLD time-series after removing scans: ', fMRI_BOLD.shape

# now, save all BOLD timeseries to a single file 
np.save(BOLD_file, fMRI_BOLD)

plt.figure()

plt.suptitle('SIMULATED SYNAPTIC SIGNAL')

# plot synaptic time-series for all ROIs
for idx, roi in enumerate(syn):
    plt.plot(syn[idx], linewidth=1.0, color='black')

plt.figure()

plt.suptitle('SIMULATED fMRI BOLD SIGNAL')

# plot BOLD time-series for all ROIs
for idx, roi in enumerate(fMRI_BOLD):
    plt.plot(fMRI_BOLD[idx], linewidth=1.0, color='black')

# calculate correlation matrix
corr_mat = np.corrcoef(fMRI_BOLD)
print 'Number of ROIs: ', fMRI_BOLD.shape
print 'Shape of correlation matrix: ', corr_mat.shape

# save cross-correlation matrix to an npy python file
np.save(xcorr_file, corr_mat)

#initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)

# plot correlation matrix as a heatmap
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(corr_mat, vmin=-1, vmax=1.0, interpolation='nearest', cmap=cmap)
ax.grid(False)
plt.colorbar(cax)

# initialize new figure for plotting histogram
fig = plt.figure()
ax = fig.add_subplot(111)

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(corr_mat.shape[1], k=0)
mask = np.transpose(mask)

# apply mask to empirical FC matrix
corr_mat = np.ma.array(corr_mat, mask=mask)    # mask out upper triangle

# flatten the numpy cross-correlation matrix
corr_mat = np.ma.ravel(corr_mat)

# remove masked elements from cross-correlation matrix
corr_mat = np.ma.compressed(corr_mat)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')

# calculate and print kurtosis
print 'Fishers kurtosis: ', kurtosis(corr_mat, fisher=True)
print 'Skewness: ', skew(corr_mat)
    
# Show the plots on the screen
plt.show()
