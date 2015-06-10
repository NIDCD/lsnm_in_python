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
#   This file (simulate_Wilson_Cowan_Brain_998_nodes.py) was created on 04/29/15,
#   based on 'generate_region_demo_data.py' by Stuart A. Knock and
#            'region_deterministic_bnm_wc.py' by Paula Sanz-Leon
#
#   This program makes use of The Virtual Brain library toolbox, downloaded
#   from the TVB GitHub page.
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on June 9 2015  
# **************************************************************************/
#
# simulate_Wilson_Cowan_Brain_998_nodes.py
#
# Simulates resting brain activity using Wilson Cowan model and 998 nodes, as
# seen in Hagmann et al (2008).
# The activation of state variable E and I are collected at each time step
# of the simulation and are written out to a data file that can be accessed
# later.
#

from tvb.simulator.lab import *

import scipy.spatial.distance as ds

import numpy as np

# white matter transmission speed in mm/ms
speed = 4.0

# define length of simulation in ms
simulation_length = 1

# define global coupling strength as in Sanz-Leon (2015) Neuroimage paper
# figure 17 3rd column 3rd row
global_coupling_strength = 0.0042


# define the population model to be used and state variables to be collected.
# the parameters below were taken from in Sanz-Leon et al (2015), table 11,
# case 'a' 
WC = models.WilsonCowan(variables_of_interest=['E','I'],
                        r_i=1, r_e=1, k_e=1, k_i=1, tau_e=10, tau_i=10,
                        c_ee=12, c_ei=4, c_ie=13, c_ii=11, alpha_e=1, alpha_i=1,
                        a_e=1.2, a_i=1, b_e=2.8, b_i=4, c_e=1, c_i=1,
                        P=0, Q=0)

# Define connectivity to be used (998 ROI matrix from TVB demo set)
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")

# Define the transmission speed of white matter tracts (4 mm/ms)
white_matter.speed = numpy.array([speed])

# Define the coupling function between white matter tracts and brain regions
white_matter_coupling = coupling.Linear(a=global_coupling_strength)

#Initialise an Integrator
# set numpy's seed
my_seed = 13
my_random_state = numpy.random.RandomState(my_seed)
hiss = noise.Additive(nsig=0.08)
heunint = integrators.EulerStochastic(dt=2 ** -2, noise=hiss)
heunint.configure()

# Define a monitor to be used (i.e., simulated data to be collected)
what_to_watch = monitors.SubSample(period=5.0)

# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model=WC, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch)

sim.configure()

# Run the simulation
raw_data = []
raw_time = []
for raw in sim(simulation_length=simulation_length):
    if raw is not None:
        raw_time.append(raw[0][0]) 
        raw_data.append(raw[0][1])

# Convert data list to a numpy array
RawData = numpy.array(raw_data)

# write output dimension to the console
print RawData.shape

print white_matter.centres

d = ds.cdist([(18, -88, 8)], white_matter.centres, 'euclidean')

closest = d[0].argmin()
print white_matter.centres[closest]

# Save the array to a file for future use
FILE_NAME = "wilson_cowan_brain_998_nodes.npy"
numpy.save(FILE_NAME, RawData)
