#!/usr/bin/python
#
# The following script recreates a Resting State Experiment.
#
# The total number of timesteps is 200 = 200*0.1 = 20 ms
#
# We assume 1 timestep = 0.1 ms
#
# TMS effects are applied: ON (3 ms = 30 timesteps), OFF (3.5ms = 35 timesteps)
#
# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 0.1 ms
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

def tms_ml23_on(modules, script_params):

    """
    injects TMS current into M1

    """
    modules['tms1'][8][0][0][0] = script_params[1]


    
def tms_ml23_off(modules, script_params):

    """
    stops TMS current injection into M1

    """
    modules['tms1'][8][0][0][0] = script_params[0]

def tms_ml5_on(modules, script_params):

    """
    injects TMS current into M1

    """
    modules['tms2'][8][0][0][0] = script_params[1]


    
def tms_ml5_off(modules, script_params):

    """
    stops TMS current injection into M1

    """
    modules['tms2'][8][0][0][0] = script_params[0]


    
# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
simulation_events = {

    '40': tms_ml23_on,
    '41': tms_ml23_off,
    '55': tms_ml5_on,
    '56': tms_ml5_off,
    '65': tms_ml5_on,
    '66': tms_ml5_off,
    '75': tms_ml5_on,
    '76': tms_ml5_off,

}


##- EoF -##
