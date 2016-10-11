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
#   Last updated by Antonio Ulloa on September 27 2016
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
# The input data (synaptic activities) and the output (BOLD time-series) are numpy arrays
# with columns in the following order:
#
# V1 ROI (right hemisphere, includes LSNM units and TVB nodes) 
# V4 ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (right hemisphere, includes LSNM units and TVB nodes)
# FS ROI (right hemisphere, includes LSNM units and TVB nodes)
# D1 ROI (right hemisphere, includes LSNM units and TVB nodes)
# D2 ROI (right hemisphere, includes LSNM units and TVB nodes)
# FR ROI (right hemisphere, includes LSNM units and TVB nodes)
# IT ROI (left hemisphere, contains only  TVB nodes)
#
#
# Finally, the cross-correlation matrix is displayed.


import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint

from matplotlib import cm as CM

# define the name of the input file where the synaptic activities are stored
SYN_file  = 'synaptic_in_66_ROIs.npy'

# define the name of the output file where the BOLD timeseries will be stored
BOLD_file = 'bold_balloon_66_regions.npy'

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
v1_syn = np.delete(syn[0], 0)
v4_syn = np.delete(syn[1], 0)
it_syn = np.delete(syn[2], 0)
fs_syn = np.delete(syn[3], 0)
d1_syn = np.delete(syn[4], 0)
d2_syn = np.delete(syn[5], 0)
fr_syn = np.delete(syn[6], 0)
lit_syn = np.delete(syn[7], 0)

# extract the synaptic activities corresponding to each ROI, and normalize to (0,1):
#v1_syn = np.append(np.ones(320)*0.4, (v1_syn-v1_syn.min()) / (v1_syn.max() - v1_syn.min()))
#v4_syn = np.append(np.ones(320)*0.4, (v4_syn-v4_syn.min()) / (v4_syn.max() - v4_syn.min()))
#it_syn = np.append(np.ones(320)*0.4, (it_syn-it_syn.min()) / (it_syn.max() - it_syn.min()))
#fs_syn = np.append(np.ones(320)*0.4, (fs_syn-fs_syn.min()) / (fs_syn.max() - fs_syn.min()))
#d1_syn = np.append(np.ones(320)*0.4, (d1_syn-d1_syn.min()) / (d1_syn.max() - d1_syn.min()))
#d2_syn = np.append(np.ones(320)*0.4, (d2_syn-d2_syn.min()) / (d2_syn.max() - d2_syn.min()))
#fr_syn = np.append(np.ones(320)*0.4, (fr_syn-fr_syn.min()) / (fr_syn.max() - fr_syn.min()))
#lit_syn= np.append(np.ones(320)*0.4, (lit_syn-lit_syn.min()) / (lit_syn.max() - lit_syn.min()))

v1_syn = (v1_syn-v1_syn.min()) / (v1_syn.max() - v1_syn.min())
v4_syn = (v4_syn-v4_syn.min()) / (v4_syn.max() - v4_syn.min())
it_syn = (it_syn-it_syn.min()) / (it_syn.max() - it_syn.min())
fs_syn = (fs_syn-fs_syn.min()) / (fs_syn.max() - fs_syn.min())
d1_syn = (d1_syn-d1_syn.min()) / (d1_syn.max() - d1_syn.min())
d2_syn = (d2_syn-d2_syn.min()) / (d2_syn.max() - d2_syn.min())
fr_syn = (fr_syn-fr_syn.min()) / (fr_syn.max() - fr_syn.min())
lit_syn= (lit_syn-lit_syn.min()) / (lit_syn.max() - lit_syn.min())


# Extract number of timesteps from one of the synaptic activity arrays
synaptic_timesteps = v1_syn.size
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
y_0_v1 = [s, f, v, q]      
y_0_v4 = [s, f, v, q]      
y_0_it = [s, f, v, q]      
y_0_fs = [s, f, v, q]      
y_0_d1 = [s, f, v, q]      
y_0_d2 = [s, f, v, q]      
y_0_fr = [s, f, v, q]      
y_0_lit= [s, f, v, q]

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
state_v1 = odeint(balloon_function, y_0_v1, t, args=(v1_syn,) )
state_v4 = odeint(balloon_function, y_0_v4, t, args=(v4_syn,) )
state_it = odeint(balloon_function, y_0_it, t, args=(it_syn,) )
state_d1 = odeint(balloon_function, y_0_d1, t, args=(d1_syn,) )
state_d2 = odeint(balloon_function, y_0_d2, t, args=(d2_syn,) )
state_fs = odeint(balloon_function, y_0_fs, t, args=(fs_syn,) )
state_fr = odeint(balloon_function, y_0_fr, t, args=(fr_syn,) )
state_lit= odeint(balloon_function, y_0_lit,t, args=(lit_syn,))

# Unpack the state variables used in the BOLD model
s_v1 = state_v1[:, 0]
f_v1 = state_v1[:, 1]
v_v1 = state_v1[:, 2]
q_v1 = state_v1[:, 3]        

s_v4 = state_v4[:, 0]
f_v4 = state_v4[:, 1]
v_v4 = state_v4[:, 2]
q_v4 = state_v4[:, 3]        

s_it = state_it[:, 0]
f_it = state_it[:, 1]
v_it = state_it[:, 2]
q_it = state_it[:, 3]        

s_fs = state_fs[:, 0]
f_fs = state_fs[:, 1]
v_fs = state_fs[:, 2]
q_fs = state_fs[:, 3]        

s_d1 = state_d1[:, 0]
f_d1 = state_d1[:, 1]
v_d1 = state_d1[:, 2]
q_d1 = state_d1[:, 3]        

s_d2 = state_d2[:, 0]
f_d2 = state_d2[:, 1]
v_d2 = state_d2[:, 2]
q_d2 = state_d2[:, 3]        

s_fr = state_fr[:, 0]
f_fr = state_fr[:, 1]
v_fr = state_fr[:, 2]
q_fr = state_fr[:, 3]        

s_lit= state_lit[:,0]
f_lit= state_lit[:,1]
v_lit= state_lit[:,2]
q_lit= state_lit[:,3]

# now, we need to calculate BOLD signal at each timestep, based on v and q obtained from solving
# balloon model ODE above.
v1_BOLD = np.array(V_0 * (k1 * (1. - q_v1) + k2 * (1. - q_v1 / v_v1) + k3 * (1. - v_v1)) )
v4_BOLD = np.array(V_0 * (k1 * (1. - q_v4) + k2 * (1. - q_v4 / v_v4) + k3 * (1. - v_v4)) )
it_BOLD = np.array(V_0 * (k1 * (1. - q_it) + k2 * (1. - q_it / v_it) + k3 * (1. - v_it)) )
fs_BOLD = np.array(V_0 * (k1 * (1. - q_fs) + k2 * (1. - q_fs / v_fs) + k3 * (1. - v_fs)) )
d1_BOLD = np.array(V_0 * (k1 * (1. - q_d1) + k2 * (1. - q_d1 / v_d1) + k3 * (1. - v_d1)) )
d2_BOLD = np.array(V_0 * (k1 * (1. - q_d2) + k2 * (1. - q_d2 / v_d2) + k3 * (1. - v_d2)) )
fr_BOLD = np.array(V_0 * (k1 * (1. - q_fr) + k2 * (1. - q_fr / v_fr) + k3 * (1. - v_fr)) )
lit_BOLD= np.array(V_0 * (k1 * (1. - q_lit)+ k2 * (1. - q_lit/v_lit) + k3 * (1. - v_lit)) )

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
v1_BOLD = v1_BOLD[scanning_timescale]
v4_BOLD = v4_BOLD[scanning_timescale]
it_BOLD = it_BOLD[scanning_timescale]
d1_BOLD = d1_BOLD[scanning_timescale]
d2_BOLD = d2_BOLD[scanning_timescale]
fs_BOLD = fs_BOLD[scanning_timescale]
fr_BOLD = fr_BOLD[scanning_timescale]
lit_BOLD=lit_BOLD[scanning_timescale]

print 'Size of BOLD arrays before deleting scans: ', v1_BOLD.size

# now we are going to remove the first trial
# estimate how many 'synaptic ticks' there are in each trial
#synaptic_ticks = Ttrial/Ti
# estimate how many 'MR ticks' there are in each trial
#mr_ticks = round(Ttrial/Tr)

# remove first few scans from BOLD signal array and from BOLD timescale array
mr_time = np.delete(mr_time, np.arange(scans_to_remove))
v1_BOLD = np.delete(v1_BOLD, np.arange(scans_to_remove))
v4_BOLD = np.delete(v4_BOLD, np.arange(scans_to_remove))
it_BOLD = np.delete(it_BOLD, np.arange(scans_to_remove))
d1_BOLD = np.delete(d1_BOLD, np.arange(scans_to_remove))
d2_BOLD = np.delete(d2_BOLD, np.arange(scans_to_remove))
fs_BOLD = np.delete(fs_BOLD, np.arange(scans_to_remove))
fr_BOLD = np.delete(fr_BOLD, np.arange(scans_to_remove))
lit_BOLD= np.delete(lit_BOLD,np.arange(scans_to_remove))

# round of mr time for display purposes
mr_time = np.round(mr_time, decimals=0)

# create a numpy array of timeseries
lsnm_BOLD = np.array([v1_BOLD, v4_BOLD, it_BOLD,
                      fs_BOLD, d1_BOLD, d2_BOLD, fr_BOLD])

print 'Size of BOLD time-series after removing scans: ', v1_BOLD.size

# now, save all BOLD timeseries to a single file 
np.save(BOLD_file, lsnm_BOLD)

# increase font size for display purposes
plt.rcParams.update({'font.size': 20})

# Set up figure to plot synaptic activity
plt.figure()

plt.suptitle('SIMULATED SYNAPTIC ACTIVITY')

# Plot synaptic activities
plt.plot(t, syn[0], linewidth=3.0, color='yellow')
plt.plot(t, syn[1], linewidth=3.0, color='blue')
plt.plot(t, syn[2], linewidth=3.0, color='red')
plt.gca().set_axis_bgcolor('black')

# Set up separate figures to plot fMRI BOLD signal
plt.figure()

plt.suptitle('SIMULATED fMRI BOLD SIGNAL')

plt.plot(mr_time, v1_BOLD, linewidth=3.0, color='yellow')
plt.gca().set_axis_bgcolor('black')

#plt.figure()

#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN V4')

plt.plot(mr_time, v4_BOLD, linewidth=3.0, color='green')
#plt.gca().set_axis_bgcolor('black')

#plt.figure()
#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN IT')

plt.plot(mr_time, it_BOLD, linewidth=3.0, color='blue')
plt.gca().set_axis_bgcolor('black')

#plt.figure()
#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN D1')

plt.plot(mr_time, fs_BOLD, linewidth=3.0, color='orange')

plt.plot(mr_time, d1_BOLD, linewidth=3.0, color='red')
#plt.gca().set_axis_bgcolor('black')

plt.plot(mr_time, d2_BOLD, linewidth=3.0, color='pink')
#plt.gca().set_axis_bgcolor('black')

plt.plot(mr_time, fr_BOLD, linewidth=3.0, color='purple')

#plt.figure()
#plt.suptitle('SIMULATED fMRI BOLD SIGNAL IN LIT')

#plt.plot(lit_BOLD, linewidth=3.0, color='pink')
#plt.gca().set_axis_bgcolor('black')

# calculate correlation matrix
corr_mat = np.corrcoef(lsnm_BOLD)

# prepare labels
labels = ['V1/V2', 'V4', 'IT', 'FS', 'D1', 'D2', 'FR']

#initialize new figure for correlations
fig = plt.figure()
ax = fig.add_subplot(111)



# plot correlation matrix as a heatmap
mask = np.tri(corr_mat.shape[0], k=-1)
mask = np.transpose(mask)
print mask
corr_mat = np.ma.array(corr_mat, mask=mask)          # mask out the lower triangle
cmap = CM.get_cmap('jet', 10)
cmap.set_bad('w')
cax = ax.imshow(corr_mat, interpolation='nearest', cmap=cmap)
ax.grid(False)
plt.colorbar(cax)

# display labels for brain regions
ax.set_xticklabels(['']+labels, minor=False)
ax.set_yticklabels(['']+labels, minor=False)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False


# Show the plots on the screen
plt.show()
