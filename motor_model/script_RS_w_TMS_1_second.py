#!/usr/bin/python
#
# The following script recreates a Resting State Experiment.
#
# The total number of timesteps is 200 = 1 second
#
# We assume 1 timestep = 5 ms, as in Horwitz et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
#
# TMS effects are applied: ON (50 ms = 10 timesteps), OFF (150ms = 30 timesteps)
#
# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = 200

script_params = [0.0, 1.0]

def tms_on(modules, script_params):

    """
    injects TMS current into M1

    """

    modules['etms'][8][0][0][0] = script_params[1]

def tms_off(modules, script_params):

    """
    stops TMS current injection into M1

    """

    modules['etms'][8][0][0][0] = script_params[0]

    
# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
simulation_events = {

    '100': tms_on,
    '110': tms_off,

}


##- EoF -##
