#!/usr/bin/python
#
# The following script recreates a Resting State Experiment.
#
# The total number of timesteps is 6000 = 30 seconds
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
LSNM_simulation_time = 6000

script_params = [0.0, 0.1]

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

    '1010': tms_on,
    '1020': tms_off,
    '1050': tms_on,
    '1060': tms_off,
    '1090': tms_on,
    '1100': tms_off,
    '1130': tms_on,
    '1140': tms_off,
    '1170': tms_on,
    '1180': tms_off,
    '1210': tms_on,
    '1220': tms_off,
    '1250': tms_on,
    '1260': tms_off,
    '1290': tms_on,
    '1300': tms_off,

}


##- EoF -##
