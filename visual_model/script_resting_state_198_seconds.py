#!/usr/bin/python
#
# The following script recreates a Resting State Experiment.
#
# The total number of timesteps is 39600 = 198 seconds
#
# We assume 1 timestep = 5 ms, as in Horwitz et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
#
# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = 39600

# define our attention parameter to a low setting since we are running a resting
# state simulation. We will use 0.05, which is the same attention level that we
# use for passive viewing during DMS task simulations.
modules['atts'][8][0][0][0] = 0.05

# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
simulation_events = {        

}


##- EoF -##
