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
#   This file (compute_BOLD_balloon_66_regions.py) was created on September 27, 2016.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on January 3 2017
#
# **************************************************************************/

# compute_BOLD_balloon_66_regions.py
#
# Calculate and plot fMRI BOLD signal, using the
# Balloon model, as described by Stephan et al (2007)
# and Friston et al (2000), and
# the BOLD signal model, as described by Stephan et al (2007) and
# Friston et al (2000).
# Parameters for both were taken from Friston et al (2000) and they were
# estimated using a 2T scanner, with a TR of 2 seconds, 
# 
# ... using data from visual delay-match-to-sample simulation, 66 regions from
# Hagmann's connectome.
# It also saves the BOLD timeseries for each and all modules in a python data file
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
SYN_file  = 'synaptic_in_66_ROIs.npy'

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'bold_balloon_66_regions.npy'

# define the name of the output file where the cross-correlation matrix will
# be stored
xcorr_file = 'xcorr_matrix_66_regions.npy'

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
y_0_33 = [s, f, v, q]      
y_0_34 = [s, f, v, q]      
y_0_35 = [s, f, v, q]      
y_0_36 = [s, f, v, q]      
y_0_37 = [s, f, v, q]      
y_0_38 = [s, f, v, q]      
y_0_39 = [s, f, v, q]      
y_0_40 = [s, f, v, q]      
y_0_41 = [s, f, v, q]      
y_0_42 = [s, f, v, q]      
y_0_43 = [s, f, v, q]      
y_0_44 = [s, f, v, q]      
y_0_45 = [s, f, v, q]      
y_0_46 = [s, f, v, q]      
y_0_47 = [s, f, v, q]      
y_0_48 = [s, f, v, q]      
y_0_49 = [s, f, v, q]      
y_0_50 = [s, f, v, q]      
y_0_51 = [s, f, v, q]      
y_0_52 = [s, f, v, q]      
y_0_53 = [s, f, v, q]      
y_0_54 = [s, f, v, q]      
y_0_55 = [s, f, v, q]      
y_0_56 = [s, f, v, q]      
y_0_57 = [s, f, v, q]      
y_0_58 = [s, f, v, q]      
y_0_59 = [s, f, v, q]      
y_0_60 = [s, f, v, q]      
y_0_61 = [s, f, v, q]      
y_0_62 = [s, f, v, q]
y_0_63 = [s, f, v, q]      
y_0_64 = [s, f, v, q]      
y_0_65 = [s, f, v, q]      


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
state_33 = odeint(balloon_function, y_0_33, t, args=(syn[33],) )
state_34 = odeint(balloon_function, y_0_34, t, args=(syn[34],) )
state_35 = odeint(balloon_function, y_0_35, t, args=(syn[35],) )
state_36 = odeint(balloon_function, y_0_36, t, args=(syn[36],) )
state_37 = odeint(balloon_function, y_0_37, t, args=(syn[37],) )
state_38 = odeint(balloon_function, y_0_38, t, args=(syn[38],) )
state_39 = odeint(balloon_function, y_0_39, t, args=(syn[39],) )
state_40 = odeint(balloon_function, y_0_40, t, args=(syn[40],) )
state_41 = odeint(balloon_function, y_0_41, t, args=(syn[41],) )
state_42 = odeint(balloon_function, y_0_42, t, args=(syn[42],) )
state_43 = odeint(balloon_function, y_0_43, t, args=(syn[43],) )
state_44 = odeint(balloon_function, y_0_44, t, args=(syn[44],) )
state_45 = odeint(balloon_function, y_0_45, t, args=(syn[45],) )
state_46 = odeint(balloon_function, y_0_46, t, args=(syn[46],) )
state_47 = odeint(balloon_function, y_0_47, t, args=(syn[47],) )
state_48 = odeint(balloon_function, y_0_48, t, args=(syn[48],) )
state_49 = odeint(balloon_function, y_0_49, t, args=(syn[49],) )
state_50 = odeint(balloon_function, y_0_50, t, args=(syn[50],) )
state_51 = odeint(balloon_function, y_0_51, t, args=(syn[51],) )
state_52 = odeint(balloon_function, y_0_52, t, args=(syn[52],) )
state_53 = odeint(balloon_function, y_0_53, t, args=(syn[53],) )
state_54 = odeint(balloon_function, y_0_54, t, args=(syn[54],) )
state_55 = odeint(balloon_function, y_0_55, t, args=(syn[55],) )
state_56 = odeint(balloon_function, y_0_56, t, args=(syn[56],) )
state_57 = odeint(balloon_function, y_0_57, t, args=(syn[57],) )
state_58 = odeint(balloon_function, y_0_58, t, args=(syn[58],) )
state_59 = odeint(balloon_function, y_0_59, t, args=(syn[59],) )
state_60 = odeint(balloon_function, y_0_60, t, args=(syn[60],) )
state_61 = odeint(balloon_function, y_0_61, t, args=(syn[61],) )
state_62 = odeint(balloon_function, y_0_62, t, args=(syn[62],) )
state_63 = odeint(balloon_function, y_0_63, t, args=(syn[63],) )
state_64 = odeint(balloon_function, y_0_64, t, args=(syn[64],) )
state_65 = odeint(balloon_function, y_0_65, t, args=(syn[65],) )

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

s_33 = state_33[:, 0]
f_33 = state_33[:, 1]
v_33 = state_33[:, 2]
q_33 = state_33[:, 3]        

s_34 = state_34[:, 0]
f_34 = state_34[:, 1]
v_34 = state_34[:, 2]
q_34 = state_34[:, 3]        

s_35 = state_35[:, 0]
f_35 = state_35[:, 1]
v_35 = state_35[:, 2]
q_35 = state_35[:, 3]        

s_36 = state_36[:, 0]
f_36 = state_36[:, 1]
v_36 = state_36[:, 2]
q_36 = state_36[:, 3]        

s_37 = state_37[:,0]
f_37 = state_37[:,1]
v_37 = state_37[:,2]
q_37 = state_37[:,3]

s_38 = state_38[:, 0]
f_38 = state_38[:, 1]
v_38 = state_38[:, 2]
q_38 = state_38[:, 3]        

s_39 = state_39[:, 0]
f_39 = state_39[:, 1]
v_39 = state_39[:, 2]
q_39 = state_39[:, 3]        

s_40 = state_40[:, 0]
f_40 = state_40[:, 1]
v_40 = state_40[:, 2]
q_40 = state_40[:, 3]        

s_41 = state_41[:, 0]
f_41 = state_41[:, 1]
v_41 = state_41[:, 2]
q_41 = state_41[:, 3]        

s_42 = state_42[:, 0]
f_42 = state_42[:, 1]
v_42 = state_42[:, 2]
q_42 = state_42[:, 3]        

s_43 = state_43[:, 0]
f_43 = state_43[:, 1]
v_43 = state_43[:, 2]
q_43 = state_43[:, 3]        

s_44 = state_44[:, 0]
f_44 = state_44[:, 1]
v_44 = state_44[:, 2]
q_44 = state_44[:, 3]        

s_45= state_45[:,0]
f_45= state_45[:,1]
v_45= state_45[:,2]
q_45= state_45[:,3]

s_46 = state_46[:, 0]
f_46 = state_46[:, 1]
v_46 = state_46[:, 2]
q_46 = state_46[:, 3]        

s_47 = state_47[:, 0]
f_47 = state_47[:, 1]
v_47 = state_47[:, 2]
q_47 = state_47[:, 3]        

s_48 = state_48[:, 0]
f_48 = state_48[:, 1]
v_48 = state_48[:, 2]
q_48 = state_48[:, 3]        

s_49 = state_49[:, 0]
f_49 = state_49[:, 1]
v_49 = state_49[:, 2]
q_49 = state_49[:, 3]        

s_50 = state_50[:, 0]
f_50 = state_50[:, 1]
v_50 = state_50[:, 2]
q_50 = state_50[:, 3]        

s_51 = state_51[:, 0]
f_51 = state_51[:, 1]
v_51 = state_51[:, 2]
q_51 = state_51[:, 3]        

s_52 = state_52[:,0]
f_52 = state_52[:,1]
v_52 = state_52[:,2]
q_52 = state_52[:,3]

s_53 = state_53[:, 0]
f_53 = state_53[:, 1]
v_53 = state_53[:, 2]
q_53 = state_53[:, 3]        

s_54 = state_54[:, 0]
f_54 = state_54[:, 1]
v_54 = state_54[:, 2]
q_54 = state_54[:, 3]        

s_55 = state_55[:, 0]
f_55 = state_55[:, 1]
v_55 = state_55[:, 2]
q_55 = state_55[:, 3]        

s_56 = state_56[:, 0]
f_56 = state_56[:, 1]
v_56 = state_56[:, 2]
q_56 = state_56[:, 3]        

s_57 = state_57[:, 0]
f_57 = state_57[:, 1]
v_57 = state_57[:, 2]
q_57 = state_57[:, 3]        

s_58 = state_58[:, 0]
f_58 = state_58[:, 1]
v_58 = state_58[:, 2]
q_58 = state_58[:, 3]        

s_59= state_59[:,0]
f_59= state_59[:,1]
v_59= state_59[:,2]
q_59= state_59[:,3]

s_60 = state_60[:, 0]
f_60 = state_60[:, 1]
v_60 = state_60[:, 2]
q_60 = state_60[:, 3]        

s_61 = state_61[:, 0]
f_61 = state_61[:, 1]
v_61 = state_61[:, 2]
q_61 = state_61[:, 3]        

s_62 = state_62[:, 0]
f_62 = state_62[:, 1]
v_62 = state_62[:, 2]
q_62 = state_62[:, 3]        

s_63 = state_63[:, 0]
f_63 = state_63[:, 1]
v_63 = state_63[:, 2]
q_63 = state_63[:, 3]        

s_64 = state_64[:, 0]
f_64 = state_64[:, 1]
v_64 = state_64[:, 2]
q_64 = state_64[:, 3]        

s_65 = state_65[:, 0]
f_65 = state_65[:, 1]
v_65 = state_65[:, 2]
q_65 = state_65[:, 3]        

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
BOLD_33 = np.array(V_0 * (k1 * (1. - q_33) + k2 * (1. - q_33 / v_33) + k3 * (1. - v_33)) )
BOLD_34 = np.array(V_0 * (k1 * (1. - q_34) + k2 * (1. - q_34 / v_34) + k3 * (1. - v_34)) )
BOLD_35 = np.array(V_0 * (k1 * (1. - q_35) + k2 * (1. - q_35 / v_35) + k3 * (1. - v_35)) )
BOLD_36 = np.array(V_0 * (k1 * (1. - q_36) + k2 * (1. - q_36 / v_36) + k3 * (1. - v_36)) )
BOLD_37 = np.array(V_0 * (k1 * (1. - q_37) + k2 * (1. - q_37 / v_37) + k3 * (1. - v_37)) )
BOLD_38 = np.array(V_0 * (k1 * (1. - q_38) + k2 * (1. - q_38 / v_38) + k3 * (1. - v_38)) )
BOLD_39 = np.array(V_0 * (k1 * (1. - q_39) + k2 * (1. - q_39 / v_39) + k3 * (1. - v_39)) )
BOLD_40 = np.array(V_0 * (k1 * (1. - q_40) + k2 * (1. - q_40 / v_40) + k3 * (1. - v_40)) )
BOLD_41 = np.array(V_0 * (k1 * (1. - q_41) + k2 * (1. - q_41 / v_41) + k3 * (1. - v_41)) )
BOLD_42 = np.array(V_0 * (k1 * (1. - q_42) + k2 * (1. - q_42 / v_42) + k3 * (1. - v_42)) )
BOLD_43 = np.array(V_0 * (k1 * (1. - q_43) + k2 * (1. - q_43 / v_43) + k3 * (1. - v_43)) )
BOLD_44 = np.array(V_0 * (k1 * (1. - q_44) + k2 * (1. - q_44 / v_44) + k3 * (1. - v_44)) )
BOLD_45 = np.array(V_0 * (k1 * (1. - q_45) + k2 * (1. - q_45 / v_45) + k3 * (1. - v_45)) )
BOLD_46 = np.array(V_0 * (k1 * (1. - q_46) + k2 * (1. - q_46 / v_46) + k3 * (1. - v_46)) )
BOLD_47 = np.array(V_0 * (k1 * (1. - q_47) + k2 * (1. - q_47 / v_47) + k3 * (1. - v_47)) )
BOLD_48 = np.array(V_0 * (k1 * (1. - q_48) + k2 * (1. - q_48 / v_48) + k3 * (1. - v_48)) )
BOLD_49 = np.array(V_0 * (k1 * (1. - q_49) + k2 * (1. - q_49 / v_49) + k3 * (1. - v_49)) )
BOLD_50 = np.array(V_0 * (k1 * (1. - q_50) + k2 * (1. - q_50 / v_50) + k3 * (1. - v_50)) )
BOLD_51 = np.array(V_0 * (k1 * (1. - q_51) + k2 * (1. - q_51 / v_51) + k3 * (1. - v_51)) )
BOLD_52 = np.array(V_0 * (k1 * (1. - q_52) + k2 * (1. - q_52 / v_52) + k3 * (1. - v_52)) )
BOLD_53 = np.array(V_0 * (k1 * (1. - q_53) + k2 * (1. - q_53 / v_53) + k3 * (1. - v_53)) )
BOLD_54 = np.array(V_0 * (k1 * (1. - q_54) + k2 * (1. - q_54 / v_54) + k3 * (1. - v_54)) )
BOLD_55 = np.array(V_0 * (k1 * (1. - q_55) + k2 * (1. - q_55 / v_55) + k3 * (1. - v_55)) )
BOLD_56 = np.array(V_0 * (k1 * (1. - q_56) + k2 * (1. - q_56 / v_56) + k3 * (1. - v_56)) )
BOLD_57 = np.array(V_0 * (k1 * (1. - q_57) + k2 * (1. - q_57 / v_57) + k3 * (1. - v_57)) )
BOLD_58 = np.array(V_0 * (k1 * (1. - q_58) + k2 * (1. - q_58 / v_58) + k3 * (1. - v_58)) )
BOLD_59 = np.array(V_0 * (k1 * (1. - q_59) + k2 * (1. - q_59 / v_59) + k3 * (1. - v_59)) )
BOLD_60 = np.array(V_0 * (k1 * (1. - q_60) + k2 * (1. - q_60 / v_60) + k3 * (1. - v_60)) )
BOLD_61 = np.array(V_0 * (k1 * (1. - q_61) + k2 * (1. - q_61 / v_61) + k3 * (1. - v_61)) )
BOLD_62 = np.array(V_0 * (k1 * (1. - q_62) + k2 * (1. - q_62 / v_62) + k3 * (1. - v_62)) )
BOLD_63 = np.array(V_0 * (k1 * (1. - q_63) + k2 * (1. - q_63 / v_63) + k3 * (1. - v_63)) )
BOLD_64 = np.array(V_0 * (k1 * (1. - q_64) + k2 * (1. - q_64 / v_64) + k3 * (1. - v_64)) )
BOLD_65 = np.array(V_0 * (k1 * (1. - q_65) + k2 * (1. - q_65 / v_65) + k3 * (1. - v_65)) )

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
BOLD_33 = BOLD_33[scanning_timescale]
BOLD_34 = BOLD_34[scanning_timescale]
BOLD_35 = BOLD_35[scanning_timescale]
BOLD_36 = BOLD_36[scanning_timescale]
BOLD_37 = BOLD_37[scanning_timescale]
BOLD_38 = BOLD_38[scanning_timescale]
BOLD_39 = BOLD_39[scanning_timescale]
BOLD_40 = BOLD_40[scanning_timescale]
BOLD_41 = BOLD_41[scanning_timescale]
BOLD_42 = BOLD_42[scanning_timescale]
BOLD_43 = BOLD_43[scanning_timescale]
BOLD_44 = BOLD_44[scanning_timescale]
BOLD_45 = BOLD_45[scanning_timescale]
BOLD_46 = BOLD_46[scanning_timescale]
BOLD_47 = BOLD_47[scanning_timescale]
BOLD_48 = BOLD_48[scanning_timescale]
BOLD_49 = BOLD_49[scanning_timescale]
BOLD_50 = BOLD_50[scanning_timescale]
BOLD_51 = BOLD_51[scanning_timescale]
BOLD_52 = BOLD_52[scanning_timescale]
BOLD_53 = BOLD_53[scanning_timescale]
BOLD_54 = BOLD_54[scanning_timescale]
BOLD_55 = BOLD_55[scanning_timescale]
BOLD_56 = BOLD_56[scanning_timescale]
BOLD_57 = BOLD_57[scanning_timescale]
BOLD_58 = BOLD_58[scanning_timescale]
BOLD_59 = BOLD_59[scanning_timescale]
BOLD_60 = BOLD_60[scanning_timescale]
BOLD_61 = BOLD_61[scanning_timescale]
BOLD_62 = BOLD_62[scanning_timescale]
BOLD_63 = BOLD_63[scanning_timescale]
BOLD_64 = BOLD_64[scanning_timescale]
BOLD_65 = BOLD_65[scanning_timescale]

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
BOLD_33 = np.delete(BOLD_33, np.arange(scans_to_remove))
BOLD_34 = np.delete(BOLD_34, np.arange(scans_to_remove))
BOLD_35 = np.delete(BOLD_35, np.arange(scans_to_remove))
BOLD_36 = np.delete(BOLD_36, np.arange(scans_to_remove))
BOLD_37 = np.delete(BOLD_37, np.arange(scans_to_remove))
BOLD_38 = np.delete(BOLD_38, np.arange(scans_to_remove))
BOLD_39 = np.delete(BOLD_39, np.arange(scans_to_remove))
BOLD_40 = np.delete(BOLD_40, np.arange(scans_to_remove))
BOLD_41 = np.delete(BOLD_41, np.arange(scans_to_remove))
BOLD_42 = np.delete(BOLD_42, np.arange(scans_to_remove))
BOLD_43 = np.delete(BOLD_43, np.arange(scans_to_remove))
BOLD_44 = np.delete(BOLD_44, np.arange(scans_to_remove))
BOLD_45 = np.delete(BOLD_45, np.arange(scans_to_remove))
BOLD_46 = np.delete(BOLD_46, np.arange(scans_to_remove))
BOLD_47 = np.delete(BOLD_47, np.arange(scans_to_remove))
BOLD_48 = np.delete(BOLD_48, np.arange(scans_to_remove))
BOLD_49 = np.delete(BOLD_49, np.arange(scans_to_remove))
BOLD_50 = np.delete(BOLD_50, np.arange(scans_to_remove))
BOLD_51 = np.delete(BOLD_51, np.arange(scans_to_remove))
BOLD_52 = np.delete(BOLD_52, np.arange(scans_to_remove))
BOLD_53 = np.delete(BOLD_53, np.arange(scans_to_remove))
BOLD_54 = np.delete(BOLD_54, np.arange(scans_to_remove))
BOLD_55 = np.delete(BOLD_55, np.arange(scans_to_remove))
BOLD_56 = np.delete(BOLD_56, np.arange(scans_to_remove))
BOLD_57 = np.delete(BOLD_57, np.arange(scans_to_remove))
BOLD_58 = np.delete(BOLD_58, np.arange(scans_to_remove))
BOLD_59 = np.delete(BOLD_59, np.arange(scans_to_remove))
BOLD_60 = np.delete(BOLD_60, np.arange(scans_to_remove))
BOLD_61 = np.delete(BOLD_61, np.arange(scans_to_remove))
BOLD_62 = np.delete(BOLD_62, np.arange(scans_to_remove))
BOLD_63 = np.delete(BOLD_63, np.arange(scans_to_remove))
BOLD_64 = np.delete(BOLD_64, np.arange(scans_to_remove))
BOLD_65 = np.delete(BOLD_65, np.arange(scans_to_remove))

# round of mr time for display purposes
mr_time = np.round(mr_time, decimals=0)

# create a numpy array of timeseries, using the 66 ROIs
lsnm_BOLD = np.array([BOLD_0, BOLD_1, BOLD_2, BOLD_3, BOLD_4, BOLD_5, BOLD_6, BOLD_7,
                      BOLD_8, BOLD_9, BOLD_10, BOLD_11, BOLD_12, BOLD_13, BOLD_14, BOLD_15,
                      BOLD_16, BOLD_17, BOLD_18, BOLD_19, BOLD_20, BOLD_21, BOLD_22, BOLD_23,
                      BOLD_24, BOLD_25, BOLD_26, BOLD_27, BOLD_28, BOLD_29, BOLD_30, BOLD_31,
                      BOLD_32,
                      BOLD_33, BOLD_34, BOLD_35, BOLD_36, BOLD_37, BOLD_38, BOLD_39, BOLD_40,
                      BOLD_41, BOLD_42, BOLD_43, BOLD_44, BOLD_45, BOLD_46, BOLD_47, BOLD_48,
                      BOLD_49, BOLD_50, BOLD_51, BOLD_52, BOLD_53, BOLD_54, BOLD_55, BOLD_56,
                      BOLD_57, BOLD_58, BOLD_59, BOLD_60, BOLD_61, BOLD_62, BOLD_63, BOLD_64,
                      BOLD_65 ])

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
    'rTT'  ,
    'lLOF' ,
    'lPORB',
    'lFP'  ,
    'lMOF' ,
    'lPTRI',
    'lPOPE',
    'LRMF' ,
    'lSF'  ,
    'lCMF' ,
    'lPREC',
    'lPARC',
    'lRAC' ,
    'lCAC' ,
    'lPC'  ,
    'lISTC',
    'lPSTC',
    'lSMAR',
    'lSP'  ,
    'lIP'  ,
    'lPCUN',
    'lCUN' ,
    'lPCAL',
    'lLOC' ,
    'lLING',
    'lFUS' ,
    'lPARH',
    'lENT' ,
    'lTP'  ,
    'lIT'  ,
    'lMT'  ,
    'lBSTS',
    'lST'  ,
    'lTT'
]            

#initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)

# decrease font size
#plt.rcParams.update({'font.size': 15})

# plot correlation matrix as a heatmap
#mask = np.tri(corr_mat.shape[0], k=0)
#mask = np.transpose(mask)
#corr_mat = np.ma.array(corr_mat, mask=mask)          # mask out the upper triangle
cmap = CM.get_cmap('jet', 10)
#cmap.set_bad('w')
cax = ax.imshow(corr_mat, vmin=-1, vmax=1.0, interpolation='nearest', cmap=cmap)
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
