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
#   This file (compute_BOLD_balloon_model_auditory.py) was created on April 1, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on April 1 2016
#
# **************************************************************************/

# compute_BOLD_balloon_model_auditory.py
#
# Calculate and plot fMRI BOLD signal, using the
# Balloon model, as described by Stephan et al (2007)
# and Friston et al (2000), and
# the BOLD signal model, as described by Stephan et al (2007) and
# Friston et al (2000).
# Parameters for both were taken from Friston et al (2000) and they were
# estimated using a 2T scanner, with a TR of 2 seconds, 
# 
# ... using data from auditory delay-match-to-sample simulation stored in 4
# different synaptic activity files
# It also saves the BOLD timeseries for each and all modules in a python data file
# (*.npy)
#
# The input data (synaptic activities) and the output (BOLD time-series) are numpy arrays
# with columns in the following order:
#
# A1 ROI (right hemisphere, includes LSNM units and TVB nodes) 
# A2 ROI (right hemisphere, includes LSNM units and TVB nodes)
# ST ROI (right hemisphere, includes LSNM units and TVB nodes)
# PF ROI (right hemisphere, includes LSNM units and TVB nodes)
#
# Note: only the BOLD activity of one of the synaptic activity files is computed and stored.

import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint

# define the name of the input files where the synaptic activities are stored
# we have four files: TC-PSL, Tones-PSL, TC-DMS, Tones-DMS
SYN_file_2 = 'output.TC_PSL/synaptic_in_ROI.npy'
SYN_file_3 = 'output.Tones_PSL/synaptic_in_ROI.npy'
SYN_file_4 = 'output.TC_DMS/synaptic_in_ROI.npy'
SYN_file_1 = 'output.Tones_DMS/synaptic_in_ROI.npy'

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'output.Tones_DMS/lsnm_bold_balloon.npy'

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

# calculate BOLD model coefficients, from Friston et al (2000)
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
# timestep is equivalent to 3.5 ms.
Ti = 0.0035 * 10

# Total time of scanning experiment in seconds (timesteps X 5)
T = 34.3

# Time for one complete trial in milliseconds
Ttrial = 2.8

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

# how many scans do you want to remove from beginning of BOLD timeseries?
scans_to_remove = 0

# read the input file that contains the synaptic activities of all ROIs
syn1 = np.load(SYN_file_1)
syn2 = np.load(SYN_file_2)
syn3 = np.load(SYN_file_3)
syn4 = np.load(SYN_file_4)

print 'LENGTH OF SYNAPTIC TIME-SERIES: ', syn1[0].size

# Throw away first value of each synaptic array (it is always zero)
a1_syn1 = np.delete(syn1[0], 0)
a2_syn1 = np.delete(syn1[1], 0)
st_syn1 = np.delete(syn1[2], 0)
pf_syn1 = np.delete(syn1[3], 0)
a1_syn2 = np.delete(syn2[0], 0)
a2_syn2 = np.delete(syn2[1], 0)
st_syn2 = np.delete(syn2[2], 0)
pf_syn2 = np.delete(syn2[3], 0)
a1_syn3 = np.delete(syn3[0], 0)
a2_syn3 = np.delete(syn3[1], 0)
st_syn3 = np.delete(syn3[2], 0)
pf_syn3 = np.delete(syn3[3], 0)
a1_syn4 = np.delete(syn4[0], 0)
a2_syn4 = np.delete(syn4[1], 0)
st_syn4 = np.delete(syn4[2], 0)
pf_syn4 = np.delete(syn4[3], 0)

# put together all the synaptic activity array that form part of a study, only
# for normalization purposes
# put together synaptic arrays that correspond to the same brain region
a1 = np.append(a1_syn1, [a1_syn2, a1_syn3, a1_syn4])
a2 = np.append(a2_syn1, [a2_syn2, a2_syn3, a2_syn4])
st = np.append(st_syn1, [st_syn2, st_syn3, st_syn4])
pf = np.append(pf_syn1, [pf_syn2, pf_syn3, pf_syn4])

# extract the synaptic activities corresponding to each ROI, and normalize to (0,1):
a1_syn = (a1_syn1-a1.min()) / (a1.max() - a1.min())
a2_syn = (a2_syn1-a2.min()) / (a2.max() - a2.min())
st_syn = (st_syn1-st.min()) / (st.max() - st.min())
pf_syn = (pf_syn1-pf.min()) / (pf.max() - pf.min())

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = a1_syn.size
print 'Size of synaptic arrays: ', synaptic_timesteps

# Given neural synaptic time interval and total time of scanning experiment,
# construct a numpy array of time points (data points provided in data files)
time_in_seconds = np.arange(0, T, Ti)

# Hard coded initial conditions
s = 0   # s, blood flow
f = 1.  # f, blood inflow
v = 1.  # v, venous blood volume
q = 1.  # q, deoxyhemoglobin content

# initial conditions vectors
y_0_a1 = [s, f, v, q]      
y_0_a2 = [s, f, v, q]      
y_0_st = [s, f, v, q]      
y_0_pf = [s, f, v, q]      

# generate synaptic time array
t_syn = np.arange(0, synaptic_timesteps)

# generate time array for solution
t = time_in_seconds

# Use the following lines to test the balloon BOLD model
#v1_syn[0:300] = .01          # 15 seconds do nothing
#v1_syn[300:320] = .1       # One-second stimulus
#v1_syn[320:920] = .01        # 30-second do nothing
#v1_syn[920:940] = .1       # two-second stimulus
#v1_syn[940:1540] = .01       # 30-second do nothing
#v1_syn[1540:1560]= .1      # three-second stimulus
#v1_syn[1560:2160]= .01       # 30-second do nothing
#v1_syn[2160:2180]= .1      # four-second stimulus
#v1_syn[2180:2780]= .01       # 30-second do nothing
#v1_syn[2780:2800]= .1      # five-second stimulus
#v1_syn[2800:3400]= .01       # 30-second do nothing
#v1_syn[3400:3420]= .1      # one-second stimulus
#v1_syn[3420:3959]= .01       # 30-second do nothing

# solve the ODEs for given initial conditions, parameters, and timesteps
state_a1 = odeint(balloon_function, y_0_a1, t, args=(a1_syn,) )
state_a2 = odeint(balloon_function, y_0_a2, t, args=(a2_syn,) )
state_st = odeint(balloon_function, y_0_st, t, args=(st_syn,) )
state_pf = odeint(balloon_function, y_0_pf, t, args=(pf_syn,) )

# Unpack the state variables used in the BOLD model
s_a1 = state_a1[:, 0]
f_a1 = state_a1[:, 1]
v_a1 = state_a1[:, 2]
q_a1 = state_a1[:, 3]        

s_a2 = state_a2[:, 0]
f_a2 = state_a2[:, 1]
v_a2 = state_a2[:, 2]
q_a2 = state_a2[:, 3]        

s_st = state_st[:, 0]
f_st = state_st[:, 1]
v_st = state_st[:, 2]
q_st = state_st[:, 3]        

s_pf = state_pf[:, 0]
f_pf = state_pf[:, 1]
v_pf = state_pf[:, 2]
q_pf = state_pf[:, 3]        

# now, we need to calculate BOLD signal at each timestep, based on v and q obtained from solving
# balloon model ODE above.
a1_BOLD = np.array(V_0 * (k1 * (1. - q_a1) + k2 * (1. - q_a1 / v_a1) + k3 * (1. - v_a1)) )
a2_BOLD = np.array(V_0 * (k1 * (1. - q_a2) + k2 * (1. - q_a2 / v_a2) + k3 * (1. - v_a2)) )
st_BOLD = np.array(V_0 * (k1 * (1. - q_st) + k2 * (1. - q_st / v_st) + k3 * (1. - v_st)) )
pf_BOLD = np.array(V_0 * (k1 * (1. - q_pf) + k2 * (1. - q_pf / v_pf) + k3 * (1. - v_pf)) )

# The following is for display purposes only:
# Construct a numpy array of timesteps (data points provided in data file)
# to convert from timesteps to time in seconds we do the following:
# Each simulation time-step equals 5 milliseconds
# However, we are recording only once every 10 time-steps
# Therefore, each data point in the output files represents 50 milliseconds.
# Thus, we need to multiply the datapoint times 50 ms...
# ... and divide by 1000 to convert to seconds
#t = np.linspace(0, 659*50./1000., num=660)
t = np.linspace(0, synaptic_timesteps * 35.0 / 1000., num=synaptic_timesteps)

# downsample the BOLD signal arrays to produce scan rate of 2 per second
scanning_timescale = np.arange(0, synaptic_timesteps, synaptic_timesteps / (T/Tr))
scanning_timescale = scanning_timescale.astype(int)      # don't forget to convert indices to integer
mr_time = t[scanning_timescale]
a1_BOLD = a1_BOLD[scanning_timescale]
a2_BOLD = a2_BOLD[scanning_timescale]
st_BOLD = st_BOLD[scanning_timescale]
pf_BOLD = pf_BOLD[scanning_timescale]

print 'Size of BOLD arrays: ', a1_BOLD.size

# round of mr time for display purposes
mr_time = np.round(mr_time, decimals=0)

# create a numpy array of timeseries
lsnm_BOLD = np.array([a1_BOLD, a2_BOLD, st_BOLD, pf_BOLD])

# now, save all BOLD timeseries to a single file 
np.save(BOLD_file, lsnm_BOLD)

# increase font size for display purposes
#plt.rcParams.update({'font.size': 30})

# Set up figure to plot synaptic activity
plt.figure()

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot synaptic activities
plt.plot(t, a1_syn, label='A1')
plt.plot(t, a2_syn, label='A2')
plt.plot(t, st_syn, label='ST')
plt.plot(t, pf_syn, label='PFC')

plt.legend()

# Set up separate figures to plot fMRI BOLD signal
plt.figure()

plt.suptitle('SIMULATED fMRI BOLD SIGNAL')

plt.plot(mr_time, a1_BOLD, label='A1')

plt.plot(mr_time, a2_BOLD, label='A2')

plt.plot(mr_time, st_BOLD, label='ST')

plt.plot(mr_time, pf_BOLD, label='PFC')

plt.legend()

# Show the plots on the screen
plt.show()
