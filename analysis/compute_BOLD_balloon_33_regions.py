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
#   This file (compute_BOLD_balloon_33_regions.py) was created on November 22, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on November 22 2016
#
# **************************************************************************/
#
# compute_BOLD_balloon_33_regions.py
#
# Calculate and plot fMRI BOLD signal, using the
# Balloon model, as described by Stephan et al (2007)
# and Friston et al (2000), and
# the BOLD signal model, as described by Stephan et al (2007) and
# Friston et al (2000).
# Parameters for both were taken from Friston et al (2000) and they were
# estimated using a 2T scanner, with a TR of 2 seconds, 
# 
# ... using data from visual delay-match-to-sample simulation, 33 regions from
# Hagmann's connectome (right hemisphere).
# It also saves the BOLD timeseries for each and all modules in a python data file
# (*.npy)
#
# Finally, the cross-correlation matrix is displayed and saved in a python data file.
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
SYN_file  = 'synaptic_in_33_ROIs.npy'

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'bold_balloon_33_regions.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
xcorr_file = 'xcorr_matrix_33_regions.npy'

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
T = 210
#T = 110

# Time for one complete trial in milliseconds
#Ttrial = 5.5

# the scanning happened every Tr interval below (in milliseconds). This
# is the time needed to sample hemodynamic activity to produce
# each fMRI image.
Tr = 2

# how many scans do you want to remove from beginning of BOLD timeseries?
scans_to_remove = 8

# read the input file that contains the synaptic activities of all ROIs
syn = np.load(SYN_file)

print 'LENGTH OF SYNAPTIC TIME-SERIES: ', syn[0].size

# Throw away first value of each synaptic array (it is always zero)
syn = syn[:, 1:]

# extract the synaptic activities corresponding to each ROI, and normalize to (0,1):
#v1_syn = np.append(np.ones(320)*0.4, (v1_syn-v1_syn.min()) / (v1_syn.max() - v1_syn.min()))
#v4_syn = np.append(np.ones(320)*0.4, (v4_syn-v4_syn.min()) / (v4_syn.max() - v4_syn.min()))
#it_syn = np.append(np.ones(320)*0.4, (it_syn-it_syn.min()) / (it_syn.max() - it_syn.min()))
#fs_syn = np.append(np.ones(320)*0.4, (fs_syn-fs_syn.min()) / (fs_syn.max() - fs_syn.min()))
#d1_syn = np.append(np.ones(320)*0.4, (d1_syn-d1_syn.min()) / (d1_syn.max() - d1_syn.min()))
#d2_syn = np.append(np.ones(320)*0.4, (d2_syn-d2_syn.min()) / (d2_syn.max() - d2_syn.min()))
#fr_syn = np.append(np.ones(320)*0.4, (fr_syn-fr_syn.min()) / (fr_syn.max() - fr_syn.min()))
#lit_syn= np.append(np.ones(320)*0.4, (lit_syn-lit_syn.min()) / (lit_syn.max() - lit_syn.min()))

# normalize
for idx, roi in enumerate(syn):
    syn[idx] = (syn[idx] - syn[idx].min()) / (syn[idx].max() - syn[idx].min())

# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = syn[0].size
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
y_0_0 = [s, f, v, q]      
y_0_1 = [s, f, v, q]      
y_0_2 = [s, f, v, q]      
y_0_3 = [s, f, v, q]      
y_0_4 = [s, f, v, q]      
y_0_5 = [s, f, v, q]      
y_0_6 = [s, f, v, q]      
y_0_7 = [s, f, v, q]      
y_0_8 = [s, f, v, q]      
y_0_9 = [s, f, v, q]      
y_0_10 = [s, f, v, q]      
y_0_11 = [s, f, v, q]      
y_0_12 = [s, f, v, q]      
y_0_13 = [s, f, v, q]      
y_0_14 = [s, f, v, q]      
y_0_15 = [s, f, v, q]      
y_0_16 = [s, f, v, q]      
y_0_17 = [s, f, v, q]      
y_0_18 = [s, f, v, q]      
y_0_19 = [s, f, v, q]      
y_0_20 = [s, f, v, q]      
y_0_21 = [s, f, v, q]      
y_0_22 = [s, f, v, q]      
y_0_23 = [s, f, v, q]      
y_0_24 = [s, f, v, q]      
y_0_25 = [s, f, v, q]      
y_0_26 = [s, f, v, q]      
y_0_27 = [s, f, v, q]      
y_0_28 = [s, f, v, q]      
y_0_29 = [s, f, v, q]      
y_0_30 = [s, f, v, q]      
y_0_31 = [s, f, v, q]      
y_0_32 = [s, f, v, q]


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
state_0 = odeint(balloon_function, y_0_0, t, args=(syn[0],) )
state_1 = odeint(balloon_function, y_0_1, t, args=(syn[1],) )
state_2 = odeint(balloon_function, y_0_2, t, args=(syn[2],) )
state_3 = odeint(balloon_function, y_0_3, t, args=(syn[3],) )
state_4 = odeint(balloon_function, y_0_4, t, args=(syn[4],) )
state_5 = odeint(balloon_function, y_0_5, t, args=(syn[5],) )
state_6 = odeint(balloon_function, y_0_6, t, args=(syn[6],) )
state_7 = odeint(balloon_function, y_0_7, t, args=(syn[7],) )
state_8 = odeint(balloon_function, y_0_8, t, args=(syn[8],) )
state_9 = odeint(balloon_function, y_0_9, t, args=(syn[9],) )
state_10 = odeint(balloon_function, y_0_10, t, args=(syn[10],) )
state_11 = odeint(balloon_function, y_0_11, t, args=(syn[11],) )
state_12 = odeint(balloon_function, y_0_12, t, args=(syn[12],) )
state_13 = odeint(balloon_function, y_0_13, t, args=(syn[13],) )
state_14 = odeint(balloon_function, y_0_14, t, args=(syn[14],) )
state_15 = odeint(balloon_function, y_0_15, t, args=(syn[15],) )
state_16 = odeint(balloon_function, y_0_16, t, args=(syn[16],) )
state_17 = odeint(balloon_function, y_0_17, t, args=(syn[17],) )
state_18 = odeint(balloon_function, y_0_18, t, args=(syn[18],) )
state_19 = odeint(balloon_function, y_0_19, t, args=(syn[19],) )
state_20 = odeint(balloon_function, y_0_20, t, args=(syn[20],) )
state_21 = odeint(balloon_function, y_0_21, t, args=(syn[21],) )
state_22 = odeint(balloon_function, y_0_22, t, args=(syn[22],) )
state_23 = odeint(balloon_function, y_0_23, t, args=(syn[23],) )
state_24 = odeint(balloon_function, y_0_24, t, args=(syn[24],) )
state_25 = odeint(balloon_function, y_0_25, t, args=(syn[25],) )
state_26 = odeint(balloon_function, y_0_26, t, args=(syn[26],) )
state_27 = odeint(balloon_function, y_0_27, t, args=(syn[27],) )
state_28 = odeint(balloon_function, y_0_28, t, args=(syn[28],) )
state_29 = odeint(balloon_function, y_0_29, t, args=(syn[29],) )
state_30 = odeint(balloon_function, y_0_30, t, args=(syn[30],) )
state_31 = odeint(balloon_function, y_0_31, t, args=(syn[31],) )
state_32 = odeint(balloon_function, y_0_32, t, args=(syn[32],) )

# Unpack the state variables used in the BOLD model
s_0 = state_0[:, 0]
f_0 = state_0[:, 1]
v_0 = state_0[:, 2]
q_0 = state_0[:, 3]        

s_1 = state_1[:, 0]
f_1 = state_1[:, 1]
v_1 = state_1[:, 2]
q_1 = state_1[:, 3]        

s_2 = state_2[:, 0]
f_2 = state_2[:, 1]
v_2 = state_2[:, 2]
q_2 = state_2[:, 3]        

s_3 = state_3[:, 0]
f_3 = state_3[:, 1]
v_3 = state_3[:, 2]
q_3 = state_3[:, 3]        

s_4 = state_4[:, 0]
f_4 = state_4[:, 1]
v_4 = state_4[:, 2]
q_4 = state_4[:, 3]        

s_5 = state_5[:, 0]
f_5 = state_5[:, 1]
v_5 = state_5[:, 2]
q_5 = state_5[:, 3]        

s_6 = state_6[:, 0]
f_6 = state_6[:, 1]
v_6 = state_6[:, 2]
q_6 = state_6[:, 3]        

s_7 = state_7[:,0]
f_7 = state_7[:,1]
v_7 = state_7[:,2]
q_7 = state_7[:,3]

s_8 = state_8[:, 0]
f_8 = state_8[:, 1]
v_8 = state_8[:, 2]
q_8 = state_8[:, 3]        

s_9 = state_9[:, 0]
f_9 = state_9[:, 1]
v_9 = state_9[:, 2]
q_9 = state_9[:, 3]        

s_10 = state_10[:, 0]
f_10 = state_10[:, 1]
v_10 = state_10[:, 2]
q_10 = state_10[:, 3]        

s_11 = state_11[:, 0]
f_11 = state_11[:, 1]
v_11 = state_11[:, 2]
q_11 = state_11[:, 3]        

s_12 = state_12[:, 0]
f_12 = state_12[:, 1]
v_12 = state_12[:, 2]
q_12 = state_12[:, 3]        

s_13 = state_13[:, 0]
f_13 = state_13[:, 1]
v_13 = state_13[:, 2]
q_13 = state_13[:, 3]        

s_14 = state_14[:, 0]
f_14 = state_14[:, 1]
v_14 = state_14[:, 2]
q_14 = state_14[:, 3]        

s_15= state_15[:,0]
f_15= state_15[:,1]
v_15= state_15[:,2]
q_15= state_15[:,3]

s_16 = state_16[:, 0]
f_16 = state_16[:, 1]
v_16 = state_16[:, 2]
q_16 = state_16[:, 3]        

s_17 = state_17[:, 0]
f_17 = state_17[:, 1]
v_17 = state_17[:, 2]
q_17 = state_17[:, 3]        

s_18 = state_18[:, 0]
f_18 = state_18[:, 1]
v_18 = state_18[:, 2]
q_18 = state_18[:, 3]        

s_19 = state_19[:, 0]
f_19 = state_19[:, 1]
v_19 = state_19[:, 2]
q_19 = state_19[:, 3]        

s_20 = state_20[:, 0]
f_20 = state_20[:, 1]
v_20 = state_20[:, 2]
q_20 = state_20[:, 3]        

s_21 = state_21[:, 0]
f_21 = state_21[:, 1]
v_21 = state_21[:, 2]
q_21 = state_21[:, 3]        

s_22 = state_22[:,0]
f_22 = state_22[:,1]
v_22 = state_22[:,2]
q_22 = state_22[:,3]

s_23 = state_23[:, 0]
f_23 = state_23[:, 1]
v_23 = state_23[:, 2]
q_23 = state_23[:, 3]        

s_24 = state_24[:, 0]
f_24 = state_24[:, 1]
v_24 = state_24[:, 2]
q_24 = state_24[:, 3]        

s_25 = state_25[:, 0]
f_25 = state_25[:, 1]
v_25 = state_25[:, 2]
q_25 = state_25[:, 3]        

s_26 = state_26[:, 0]
f_26 = state_26[:, 1]
v_26 = state_26[:, 2]
q_26 = state_26[:, 3]        

s_27 = state_27[:, 0]
f_27 = state_27[:, 1]
v_27 = state_27[:, 2]
q_27 = state_27[:, 3]        

s_28 = state_28[:, 0]
f_28 = state_28[:, 1]
v_28 = state_28[:, 2]
q_28 = state_28[:, 3]        

s_29= state_29[:,0]
f_29= state_29[:,1]
v_29= state_29[:,2]
q_29= state_29[:,3]

s_30 = state_30[:, 0]
f_30 = state_30[:, 1]
v_30 = state_30[:, 2]
q_30 = state_30[:, 3]        

s_31 = state_31[:, 0]
f_31 = state_31[:, 1]
v_31 = state_31[:, 2]
q_31 = state_31[:, 3]        

s_32 = state_32[:, 0]
f_32 = state_32[:, 1]
v_32 = state_32[:, 2]
q_32 = state_32[:, 3]        


# now, we need to calculate BOLD signal at each timestep, based on v and q obtained from solving
# balloon model ODE above.
BOLD_0 = np.array(V_0 * (k1 * (1. - q_0) + k2 * (1. - q_0 / v_0) + k3 * (1. - v_0)) )
BOLD_1 = np.array(V_0 * (k1 * (1. - q_1) + k2 * (1. - q_1 / v_1) + k3 * (1. - v_1)) )
BOLD_2 = np.array(V_0 * (k1 * (1. - q_2) + k2 * (1. - q_2 / v_2) + k3 * (1. - v_2)) )
BOLD_3 = np.array(V_0 * (k1 * (1. - q_3) + k2 * (1. - q_3 / v_3) + k3 * (1. - v_3)) )
BOLD_4 = np.array(V_0 * (k1 * (1. - q_4) + k2 * (1. - q_4 / v_4) + k3 * (1. - v_4)) )
BOLD_5 = np.array(V_0 * (k1 * (1. - q_5) + k2 * (1. - q_5 / v_5) + k3 * (1. - v_5)) )
BOLD_6 = np.array(V_0 * (k1 * (1. - q_6) + k2 * (1. - q_6 / v_6) + k3 * (1. - v_6)) )
BOLD_7 = np.array(V_0 * (k1 * (1. - q_7) + k2 * (1. - q_7 / v_7) + k3 * (1. - v_7)) )
BOLD_8 = np.array(V_0 * (k1 * (1. - q_8) + k2 * (1. - q_8 / v_8) + k3 * (1. - v_8)) )
BOLD_9 = np.array(V_0 * (k1 * (1. - q_9) + k2 * (1. - q_9 / v_9) + k3 * (1. - v_9)) )
BOLD_10 = np.array(V_0 * (k1 * (1. - q_10) + k2 * (1. - q_10 / v_10) + k3 * (1. - v_10)) )
BOLD_11 = np.array(V_0 * (k1 * (1. - q_11) + k2 * (1. - q_11 / v_11) + k3 * (1. - v_11)) )
BOLD_12 = np.array(V_0 * (k1 * (1. - q_12) + k2 * (1. - q_12 / v_12) + k3 * (1. - v_12)) )
BOLD_13 = np.array(V_0 * (k1 * (1. - q_13) + k2 * (1. - q_13 / v_13) + k3 * (1. - v_13)) )
BOLD_14 = np.array(V_0 * (k1 * (1. - q_14) + k2 * (1. - q_14 / v_14) + k3 * (1. - v_14)) )
BOLD_15 = np.array(V_0 * (k1 * (1. - q_15) + k2 * (1. - q_15 / v_15) + k3 * (1. - v_15)) )
BOLD_16 = np.array(V_0 * (k1 * (1. - q_16) + k2 * (1. - q_16 / v_16) + k3 * (1. - v_16)) )
BOLD_17 = np.array(V_0 * (k1 * (1. - q_17) + k2 * (1. - q_17 / v_17) + k3 * (1. - v_17)) )
BOLD_18 = np.array(V_0 * (k1 * (1. - q_18) + k2 * (1. - q_18 / v_18) + k3 * (1. - v_18)) )
BOLD_19 = np.array(V_0 * (k1 * (1. - q_19) + k2 * (1. - q_19 / v_19) + k3 * (1. - v_19)) )
BOLD_20 = np.array(V_0 * (k1 * (1. - q_20) + k2 * (1. - q_20 / v_20) + k3 * (1. - v_20)) )
BOLD_21 = np.array(V_0 * (k1 * (1. - q_21) + k2 * (1. - q_21 / v_21) + k3 * (1. - v_21)) )
BOLD_22 = np.array(V_0 * (k1 * (1. - q_22) + k2 * (1. - q_22 / v_22) + k3 * (1. - v_22)) )
BOLD_23 = np.array(V_0 * (k1 * (1. - q_23) + k2 * (1. - q_23 / v_23) + k3 * (1. - v_23)) )
BOLD_24 = np.array(V_0 * (k1 * (1. - q_24) + k2 * (1. - q_24 / v_24) + k3 * (1. - v_24)) )
BOLD_25 = np.array(V_0 * (k1 * (1. - q_25) + k2 * (1. - q_25 / v_25) + k3 * (1. - v_25)) )
BOLD_26 = np.array(V_0 * (k1 * (1. - q_26) + k2 * (1. - q_26 / v_26) + k3 * (1. - v_26)) )
BOLD_27 = np.array(V_0 * (k1 * (1. - q_27) + k2 * (1. - q_27 / v_27) + k3 * (1. - v_27)) )
BOLD_28 = np.array(V_0 * (k1 * (1. - q_28) + k2 * (1. - q_28 / v_28) + k3 * (1. - v_28)) )
BOLD_29 = np.array(V_0 * (k1 * (1. - q_29) + k2 * (1. - q_29 / v_29) + k3 * (1. - v_29)) )
BOLD_30 = np.array(V_0 * (k1 * (1. - q_30) + k2 * (1. - q_30 / v_30) + k3 * (1. - v_30)) )
BOLD_31 = np.array(V_0 * (k1 * (1. - q_31) + k2 * (1. - q_31 / v_31) + k3 * (1. - v_31)) )
BOLD_32 = np.array(V_0 * (k1 * (1. - q_32) + k2 * (1. - q_32 / v_32) + k3 * (1. - v_32)) )

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
scanning_timescale = np.linspace(0, synaptic_timesteps, num=T/Tr)
scanning_timescale = np.round(scanning_timescale)
scanning_timescale = scanning_timescale.astype(int)
mr_time = t[scanning_timescale]
BOLD_0 = BOLD_0[scanning_timescale]
BOLD_1 = BOLD_1[scanning_timescale]
BOLD_2 = BOLD_2[scanning_timescale]
BOLD_3 = BOLD_3[scanning_timescale]
BOLD_4 = BOLD_4[scanning_timescale]
BOLD_5 = BOLD_5[scanning_timescale]
BOLD_6 = BOLD_6[scanning_timescale]
BOLD_7 = BOLD_7[scanning_timescale]
BOLD_8 = BOLD_8[scanning_timescale]
BOLD_9 = BOLD_9[scanning_timescale]
BOLD_10 = BOLD_10[scanning_timescale]
BOLD_11 = BOLD_11[scanning_timescale]
BOLD_12 = BOLD_12[scanning_timescale]
BOLD_13 = BOLD_13[scanning_timescale]
BOLD_14 = BOLD_14[scanning_timescale]
BOLD_15 = BOLD_15[scanning_timescale]
BOLD_16 = BOLD_16[scanning_timescale]
BOLD_17 = BOLD_17[scanning_timescale]
BOLD_18 = BOLD_18[scanning_timescale]
BOLD_19 = BOLD_19[scanning_timescale]
BOLD_20 = BOLD_20[scanning_timescale]
BOLD_21 = BOLD_21[scanning_timescale]
BOLD_22 = BOLD_22[scanning_timescale]
BOLD_23 = BOLD_23[scanning_timescale]
BOLD_24 = BOLD_24[scanning_timescale]
BOLD_25 = BOLD_25[scanning_timescale]
BOLD_26 = BOLD_26[scanning_timescale]
BOLD_27 = BOLD_27[scanning_timescale]
BOLD_28 = BOLD_28[scanning_timescale]
BOLD_29 = BOLD_29[scanning_timescale]
BOLD_30 = BOLD_30[scanning_timescale]
BOLD_31 = BOLD_31[scanning_timescale]
BOLD_32 = BOLD_32[scanning_timescale]

print 'Size of BOLD arrays before deleting scans: ', BOLD_0.size

# now we are going to remove the first trial
# estimate how many 'synaptic ticks' there are in each trial
#synaptic_ticks = Ttrial/Ti
# estimate how many 'MR ticks' there are in each trial
#mr_ticks = round(Ttrial/Tr)

# remove first few scans from BOLD signal array and from BOLD timescale array
mr_time = np.delete(mr_time, np.arange(scans_to_remove))
BOLD_0 = np.delete(BOLD_0, np.arange(scans_to_remove))
BOLD_1 = np.delete(BOLD_1, np.arange(scans_to_remove))
BOLD_2 = np.delete(BOLD_2, np.arange(scans_to_remove))
BOLD_3 = np.delete(BOLD_3, np.arange(scans_to_remove))
BOLD_4 = np.delete(BOLD_4, np.arange(scans_to_remove))
BOLD_5 = np.delete(BOLD_5, np.arange(scans_to_remove))
BOLD_6 = np.delete(BOLD_6, np.arange(scans_to_remove))
BOLD_7 = np.delete(BOLD_7, np.arange(scans_to_remove))
BOLD_8 = np.delete(BOLD_8, np.arange(scans_to_remove))
BOLD_9 = np.delete(BOLD_9, np.arange(scans_to_remove))
BOLD_10 = np.delete(BOLD_10, np.arange(scans_to_remove))
BOLD_11 = np.delete(BOLD_11, np.arange(scans_to_remove))
BOLD_12 = np.delete(BOLD_12, np.arange(scans_to_remove))
BOLD_13 = np.delete(BOLD_13, np.arange(scans_to_remove))
BOLD_14 = np.delete(BOLD_14, np.arange(scans_to_remove))
BOLD_15 = np.delete(BOLD_15, np.arange(scans_to_remove))
BOLD_16 = np.delete(BOLD_16, np.arange(scans_to_remove))
BOLD_17 = np.delete(BOLD_17, np.arange(scans_to_remove))
BOLD_18 = np.delete(BOLD_18, np.arange(scans_to_remove))
BOLD_19 = np.delete(BOLD_19, np.arange(scans_to_remove))
BOLD_20 = np.delete(BOLD_20, np.arange(scans_to_remove))
BOLD_21 = np.delete(BOLD_21, np.arange(scans_to_remove))
BOLD_22 = np.delete(BOLD_22, np.arange(scans_to_remove))
BOLD_23 = np.delete(BOLD_23, np.arange(scans_to_remove))
BOLD_24 = np.delete(BOLD_24, np.arange(scans_to_remove))
BOLD_25 = np.delete(BOLD_25, np.arange(scans_to_remove))
BOLD_26 = np.delete(BOLD_26, np.arange(scans_to_remove))
BOLD_27 = np.delete(BOLD_27, np.arange(scans_to_remove))
BOLD_28 = np.delete(BOLD_28, np.arange(scans_to_remove))
BOLD_29 = np.delete(BOLD_29, np.arange(scans_to_remove))
BOLD_30 = np.delete(BOLD_30, np.arange(scans_to_remove))
BOLD_31 = np.delete(BOLD_31, np.arange(scans_to_remove))
BOLD_32 = np.delete(BOLD_32, np.arange(scans_to_remove))

# round of mr time for display purposes
mr_time = np.round(mr_time, decimals=0)

# create a numpy array of timeseries, using only the 33 ROIs in the right hemisphere
lsnm_BOLD = np.array([BOLD_0, BOLD_1, BOLD_2, BOLD_3, BOLD_4, BOLD_5, BOLD_6, BOLD_7,
                      BOLD_8, BOLD_9, BOLD_10, BOLD_11, BOLD_12, BOLD_13, BOLD_14, BOLD_15,
                      BOLD_16, BOLD_17, BOLD_18, BOLD_19, BOLD_20, BOLD_21, BOLD_22, BOLD_23,
                      BOLD_24, BOLD_25, BOLD_26, BOLD_27, BOLD_28, BOLD_29, BOLD_30, BOLD_31,
                      BOLD_32])

print 'Size of BOLD time-series after removing scans: ', BOLD_0.size

# now, save all BOLD timeseries to a single file 
np.save(BOLD_file, lsnm_BOLD)

# increase font size for display purposes
#plt.rcParams.update({'font.size': 20})

plt.figure()

plt.suptitle('SIMULATED SYNAPTIC SIGNAL')

# plot synaptic time-series for all ROIs
for idx, roi in enumerate(syn):
    plt.plot(syn[idx], linewidth=1.0, color='black')

plt.figure()

plt.suptitle('SIMULATED fMRI BOLD SIGNAL')

# plot BOLD time-series for all ROIs
for idx, roi in enumerate(lsnm_BOLD):
    plt.plot(mr_time, lsnm_BOLD[idx], linewidth=1.0, color='black')

# calculate correlation matrix
corr_mat = np.corrcoef(lsnm_BOLD)
print 'Number of ROIs: ', lsnm_BOLD.shape
print 'Shape of correlation matrix: ', corr_mat.shape

# save cross-correlation matrix to an npy python file
np.save(xcorr_file, corr_mat)

# prepare labels
labels =  ['rLOF',     
    'rPORB',         
    'rFP'  ,          
    'rMOF' ,          
    'rPTRI',          
    'rPOPE',          
    'rRMF' ,          
    'rSF'  ,          
    'rCMF' ,          
    'rPREC',          
    'rPARC',          
    'rRAC' ,          
    'rCAC' ,          
    'rPC'  ,          
    'rISTC',          
    'rPSTC',          
    'rSMAR',          
    'rSP'  ,          
    'rIP'  ,          
    'rPCUN',          
    'rCUN' ,          
    'rPCAL',          
    'rLOCC',          
    'rLING',          
    'rFUS' ,          
    'rPARH',          
    'rENT' ,          
    'rTP'  ,          
    'rIT'  ,          
    'rMT'  ,          
    'rBSTS',          
    'rST'  ,          
    'rTT']            

#initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)

# decrease font size
#plt.rcParams.update({'font.size': 15})

# plot correlation matrix as a heatmap
mask = np.tri(corr_mat.shape[0], k=0)
mask = np.transpose(mask)
corr_mat = np.ma.array(corr_mat, mask=mask)          # mask out the upper triangle
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w')
cax = ax.imshow(corr_mat, vmin=-1, vmax=1, interpolation='nearest', cmap=cmap)
ax.grid(False)
plt.colorbar(cax)

# change frequency of ticks to match number of ROI labels
plt.xticks(np.arange(0, len(labels)))
plt.yticks(np.arange(0, len(labels)))

# display labels for brain regions
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

# initialize new figure for plotting histogram
fig = plt.figure()
ax = fig.add_subplot(111)

# flatten the numpy cross-correlation matrix
corr_mat = np.ma.ravel(corr_mat)

# remove NaN elements from cross-correlation matrix (they are along the diagonal)
corr_mat = corr_mat[~np.isnan(corr_mat)]

# remove masked elements from cross-correlation matrix
corr_mat = np.ma.compressed(corr_mat)

# plot a histogram to show the frequency of correlations
plt.hist(corr_mat, 25)

plt.xlabel('Correlation Coefficient')
plt.ylabel('Number of occurrences')
plt.axis([-1, 1, 0, 100])

# calculate and print kurtosis
print 'Fishers kurtosis: ', kurtosis(corr_mat, fisher=True)
print 'Skewness: ', skew(corr_mat)
    
# Show the plots on the screen
plt.show()
