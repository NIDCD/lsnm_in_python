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
#   This file (sim_without_LSNM.py) was created on 8/9/15,
#   based on 'generate_region_demo_data.py' by Stuart A. Knock and
#            'region_deterministic_bnm_wc.py' by Paula Sanz-Leon,
#            'firing_rate_clamp' by Michael Marmaduke Woodman, and
#            'Evoked Responses in the Visual Cortex', by P. Sanz-Leon.
#
#   This program makes use of The Virtual Brain library toolbox, downloaded
#   from the TVB GitHub page.
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on September 13 2015  
# **************************************************************************/
#
# sim_without_LSNM.py
#
# Simulates resting brain activity using Wilson Cowan model and 998 nodes, as
# seen in Hagmann et al (2008).
# The activation of state variable E and I are collected at each time step
# of the simulation and are written out to a data file that can be accessed
# later. Synaptic activities in of E and I populations are also collected
# and saved to a file
#
# Additionnally, it finds closest TVB nodes given a set of LSNM module locations
# in Talairach coordinates

from tvb.simulator.lab import *

import scipy.spatial.distance as ds

import numpy as np

import matplotlib.pyplot as pl

neuronal_FILE = 'tvb_neuronal.npy'
synaptic_FILE = 'tvb_synaptic.npy'

# declare how many closest nodes to a given brain region will be part of that region's ROI
number_of_closest = 5

class WilsonCowanPositive(models.WilsonCowan):
    "Declares a class of Wilson-Cowan models that use the default TVB parameters but"
    "only allows positive values at integration time. In other words, it clamps state"
    "variables to > 0 when a stochastic integration is used"
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        state_variables[state_variables < 0.0] = 0.0
        state_variables[state_variables > 1.0] = 1.0
        return super(WilsonCowanPositive, self).dfun(state_variables, coupling, local_coupling)

# the following are the weights used among excitatory and inhibitory populations
# in the TVB's implementation of the Wilson-Cowan equations. These values were
# taken from the default values in the TVB source code in script "models.npy"
# The values are also in Table 11 Column a of Sanz-Leon et al (2015)
#
# We are actually re-defining the weights below to be able to calculate local
# synaptic activity at each timestep, as the TVB original source code does not
# calculate synaptic activities
w_ee = 12.0
w_ii = 11.0
w_ei =  4.0
w_ie = 13.0

# create an array to store synaptic activity for each and all TVB nodes
tvb_syna = []
# also, create an array to store electrical activity for all TVB nodes
tvb_elec = []

# number of nodes in model to be simulated (Hagmann's brain)
TVB_number_of_nodes = 998

# create and initialize array to store synaptic activity for all TVB nodes, excitatory
# and inhibitory parts.
# The synaptic activity for each node is zero at first, then it accumulates values
# (integration) during a given number of timesteps. Every number of timesteps
# (given by 'synaptic_interval'), the array below is re-initialized to zero.
current_tvb_syn = [ [0.0]*TVB_number_of_nodes for _ in range(2) ]
        
# declare an integration interval for the 'integrated' synaptic activity,
# for fMRI computation, in number of timesteps.
# The same variable is used to know how often we are going to write to
# output files
synaptic_interval = 10
        
# white matter transmission speed in mm/ms
speed = 4.0

# define length of simulation in ms
#simulation_length = 198000.0
simulation_length = 1.0

# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = simulation_length / 5.0

# Initialize timestep counter
t = 0

# define global coupling strength as in Sanz-Leon (2015) Neuroimage paper
# figure 17 3rd column 3rd row
global_coupling_strength = 0.0042

# Define connectivity to be used (998 ROI matrix from TVB demo set)
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")

# Define the transmission speed of white matter tracts (4 mm/ms)
white_matter.speed = numpy.array([speed])

# Define the coupling function between white matter tracts and brain regions
white_matter_coupling = coupling.Linear(a=global_coupling_strength)

# now, define a pulse train to be used as a stimulus to V1
white_matter.configure()
node_to_be_stimulated = 345 
stim_weights = numpy.zeros((white_matter.number_of_regions, 1))
stim_weights[node_to_be_stimulated] = numpy.array([1.0])[:, numpy.newaxis]
eqn_t = equations.PulseTrain()
eqn_t.parameters["onset"] = 2000.0 # Pulse onset in ms
eqn_t.parameters["tau"]   = 1000.0   # Pulse duration in ms
eqn_t.parameters["T"]     = 2500.  # Pulse repetition period in ms

stim = patterns.StimuliRegion(temporal = eqn_t,
                              connectivity = white_matter, 
                              weight = stim_weights)

#Initialize an Integrator
euler_int = integrators.EulerStochastic(dt=5, noise=noise.Additive(nsig=0.01))
euler_int.configure()

# Define a monitor to be used (i.e., simulated data to be collected)
what_to_watch = monitors.Raw()

# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model=WilsonCowanPositive(), connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=euler_int, monitors=what_to_watch)

sim.configure()

# create a stimulus pattern that is similar in structure to the input stimuli that
# is presented to the LSNM simulation.
# the stimuli presented is a pulse of 1000 ms duration
stimulus_pattern = np.concatenate((range(200, 401),
                                   range(700, 901),
                                   range(1300, 1501),
                                   range(1800, 2001),
                                   range(2400, 2601),
                                   range(2900, 3101),
                                   
                                   range(3500, 3701),
                                   range(4000, 4201),
                                   range(4600, 4801),
                                   range(5100, 5301),
                                   range(5700, 5901),
                                   range(6200, 6401),
                                   
                                   range(6800, 7001),
                                   range(7300, 7501),
                                   range(7900, 8101),
                                   range(8400, 8601),
                                   range(9000, 9201),
                                   range(9500, 9701),

                                   range(10100, 10301),
                                   range(10600, 10801),
                                   range(11200, 11401),
                                   range(11700, 11901),
                                   range(12300, 12501),
                                   range(12800, 13001),

                                   range(13400, 13601),
                                   range(13900, 14101),
                                   range(14500, 14701),
                                   range(15000, 15201),
                                   range(15600, 15801),
                                   range(16100, 16301),

                                   range(16700, 16901),
                                   range(17200, 17401),
                                   range(17800, 18001),
                                   range(18300, 18501),
                                   range(18900, 19101),
                                   range(19400, 19601),

                                   range(20000, 20201),
                                   range(20500, 20701),
                                   range(21100, 21301),
                                   range(21600, 21801),
                                   range(22200, 22401),
                                   range(22700, 22901),

                                   range(23300, 23501),
                                   range(23800, 24001),
                                   range(24400, 24601),
                                   range(24900, 25101),
                                   range(25500, 25701),
                                   range(26000, 26201),

                                   range(26600, 26801),
                                   range(27100, 27301),
                                   range(27700, 27901),
                                   range(28200, 28401),
                                   range(28800, 29001),
                                   range(29300, 29501),

                                   range(29900, 30101),
                                   range(30400, 30601),
                                   range(31000, 31201),
                                   range(31500, 31701),
                                   range(32100, 32301),
                                   range(32600, 32801),

                                   range(33200, 33401),
                                   range(33700, 33901),
                                   range(34300, 34501),
                                   range(34800, 35001),
                                   range(35400, 35601),
                                   range(35900, 36101),

                                   range(36500, 36701),
                                   range(37000, 37201),
                                   range(37600, 37801),
                                   range(38100, 38301),
                                   range(38700, 38901),
                                   range(39200, 39401)
                               ))

# Run the simulation
raw_data = []
raw_time = []
for raw in sim(simulation_length=simulation_length):

    print t

    # apply stimulus pattern to given node.
    if t in stimulus_pattern:
        raw[0][1][0][node_to_be_stimulated] += 0.7
    
    # the following calculates (and integrates) synaptic activity at each TVB node
    # at the current timestep
    for tvb_node in range(TVB_number_of_nodes):
        
        # rectifies or 'clamps' current tvb values to edges [0,1]
        current_tvb_neu=np.clip(raw[0][1], 0, 1)
        
        # extract TVB node numbers that are conected to TVB node above
        tvb_conn = np.nonzero(white_matter.weights[tvb_node])
        # extract the numpy array from it
        tvb_conn = tvb_conn[0]
        
        # build a numpy array of weights from TVB connections to the current TVB node
        wm = white_matter.weights[tvb_node][tvb_conn]
        
        # build a numpy array of origin TVB nodes connected to current TVB node
        tvb_origin_node = raw[0][1][0][tvb_conn]
        
        # clips node value to edges of interval [0, 1]
        tvb_origin_node = np.clip(tvb_origin_node, 0, 1)
                
        # do the following for each white matter connection to current TVB node:
        # multiply all incoming connection weights times the value of the corresponding
        # node that is sending that connection to the current TVB node
        for cxn in range(tvb_conn.size):

            # update synaptic activity in excitatory population, by multiplying each
            # incoming connection weight times the value of the node sending such
            # connection
            current_tvb_syn[0][tvb_node] += wm[cxn] * tvb_origin_node[cxn][0]

        # now, add the influence of the local (within the same node) connectivity
        # onto the synaptic activity of the current node, excitatory population
        current_tvb_syn[0][tvb_node] += w_ee * current_tvb_neu[0][tvb_node] + w_ie * current_tvb_neu[1][tvb_node]
            
        # now, update synaptic activity in inhibitory population
        # Please note that we are assuming that there are no incoming connections
        # to inhibitory nodes from other nodes (in the Virtual Brain nodes).
        # Therefore, only the local (within the same node) connections are
        # considered
        current_tvb_syn[1][tvb_node] += w_ii * current_tvb_neu[1][tvb_node] + w_ei * current_tvb_neu[0][tvb_node]
            
    # also write neural and synaptic activity of all TVB nodes to output files at
    # the current
    # time step, but ONLY IF a given number of timesteps has elapsed (integration
    # interval)
    if ((LSNM_simulation_time + t) % synaptic_interval) == 0:
        # append the current TVB node electrical activity to array
        tvb_elec.append(current_tvb_neu)
        # append current synaptic activity array to synaptic activity timeseries
        tvb_syna.append(current_tvb_syn)
        # reset TVB synaptic activity, but not TVB neuroelectrical activity
        current_tvb_syn = [ [0.0]*TVB_number_of_nodes for _ in range(2) ]

    # increase the timestep counter
    t = t + 1
        
# the following lines of code find the closest Hagmann's brain node to a given
# set of Talairach coordinates
# VISUAL MODEL TALAIRACH COORDINATES 
#d_v1 = ds.cdist([(18, -88, 8)], white_matter.centres, 'euclidean')
#closest = d_v1[0].argmin()
#print closest, white_matter.centres[closest]

#d_v4 = ds.cdist([(30, -72, -12)], white_matter.centres, 'euclidean')
#closest = d_v4[0].argmin()
#print closest, white_matter.centres[closest]

#d_it = ds.cdist([(28, -36, -8)], white_matter.centres, 'euclidean')
#closest = d_it[0].argmin()
#print closest, white_matter.centres[closest]

#d_fs= ds.cdist([(47, 19, 9)], white_matter.centres, 'euclidean')
#closest = d_fs[0].argmin()
#print closest, white_matter.centres[closest]

#d_d1= ds.cdist([(42, 26, 20)], white_matter.centres, 'euclidean')
#closest = d_d1[0].argmin()
#print closest, white_matter.centres[closest]

#d_d2= ds.cdist([(42, 39, 2)], white_matter.centres, 'euclidean')
#closest = d_d2[0].argmin()
#print closest, white_matter.centres[closest]

#d_r= ds.cdist([(29, 25, 40)], white_matter.centres, 'euclidean')
#closest = d_r[0].argmin()
#print closest, white_matter.centres[closest]

# AUDITORY MODEL TALAIRACH COORDINATES
d_a1 = ds.cdist([(48, -26, 10)], white_matter.centres, 'euclidean')
closest = d_a1[0].argsort()[:number_of_closest]
print closest, white_matter.centres[closest]

d_a2 = ds.cdist([(62, -32, 10)], white_matter.centres, 'euclidean')
#closest = d_a2[0].argmin()
closest = d_a2[0].argsort()[:number_of_closest]
print closest, white_matter.centres[closest]

d_st = ds.cdist([(59, -17, 4)], white_matter.centres, 'euclidean')
closest = d_st[0].argsort()[:number_of_closest]
print closest, white_matter.centres[closest]

d_pf= ds.cdist([(54, 9, 8)], white_matter.centres, 'euclidean')
closest = d_pf[0].argsort()[:number_of_closest]
print closest, white_matter.centres[closest]


# convert electrical and synaptic activity of TVB nodes into numpy arrays
TVB_elec = numpy.array(tvb_elec)
TVB_syna = numpy.array(tvb_syna)

# now, save the TVB electrical and synaptic activities to separate files
numpy.save(neuronal_FILE, TVB_elec)
numpy.save(synaptic_FILE, TVB_syna)
