#!/usr/bin/python
#
# The following script replicates the results of Husain et al (2004).
#
# The script has the following blocks:
#
# We have a total of _ trials in each block
#
# The number of timesteps in each trial is 800 = 4 seconds
#
# The total number of timesteps is 22000 = 110 seconds
#
# Each block is composed of 4 trials = 4400 timesteps = 22 seconds
#
# Both DMS trials and control trials are: MATCH, MISMATCH, MISMATCH, MATCH.
#
# The attention parameter in DMS trials is constant at 0.3
#
# The control trials constitute 'passive viewing': degraded shapes are presented and
# the same low attention (0.05) is used throughout the control trials
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
#                
# We present stimuli to the auditory LSNM network by manually inserting it into the LGN module
# and leaving the stimuli there for 100 timesteps (500 ms).

# define the simulation time in total number of timesteps
LSNM_simulation_time = 1700

# Define list of parameters the script is going to need to modify the LSNM neural network
# They are organized in the following order:
# [lo_att_level, hi_att_level, lo_inp_level, hi_inp_level, att_step, ri1, ri2]
script_params = [0.05, 0.3, 0.0, 1.0, 0.0, [], []]

def s1_up_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][42][0] = script_params[3]
    modules['mgns'][8][0][43][0] = script_params[3]

def s1_up_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """

    # reset previous activation
    modules['mgns'][8][0][42][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][43][0] = script_params[3]
    modules['mgns'][8][0][44][0] = script_params[3]

def s1_up_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][43][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][44][0] = script_params[3]
    modules['mgns'][8][0][45][0] = script_params[3]

def s1_up_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][44][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][45][0] = script_params[3]
    modules['mgns'][8][0][46][0] = script_params[3]

def s1_up_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][45][0] = script_params[3]
    modules['mgns'][8][0][46][0] = script_params[3]

def s1_down_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][46][0] = script_params[3]
    modules['mgns'][8][0][45][0] = script_params[3]

def s1_down_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][46][0] = script_params[3]
    modules['mgns'][8][0][45][0] = script_params[3]

def s1_down_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][46][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][45][0] = script_params[3]
    modules['mgns'][8][0][44][0] = script_params[3]

def s1_down_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][45][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][44][0] = script_params[3]
    modules['mgns'][8][0][43][0] = script_params[3]

def s1_down_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][44][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][43][0] = script_params[3]
    modules['mgns'][8][0][42][0] = script_params[3]
    
def s2_down_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][58][0] = script_params[3]
    modules['mgns'][8][0][57][0] = script_params[3]

def s2_down_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][58][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][57][0] = script_params[3]
    modules['mgns'][8][0][56][0] = script_params[3]

def s2_down_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][57][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][56][0] = script_params[3]
    modules['mgns'][8][0][55][0] = script_params[3]

def s2_down_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][56][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][54][0] = script_params[3]

def s2_down_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][54][0] = script_params[3]
    
def s2_up_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][54][0] = script_params[3]
    modules['mgns'][8][0][55][0] = script_params[3]

def s2_up_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][54][0] = script_params[3]

def s2_up_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][54][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][56][0] = script_params[3]

def s2_up_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][55][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][56][0] = script_params[3]
    modules['mgns'][8][0][57][0] = script_params[3]

def s2_up_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][56][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][57][0] = script_params[3]
    modules['mgns'][8][0][58][0] = script_params[3]

def delay_period(modules, script_params):
    
    """
    modifies neural network with delay period parameters given

    """
    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['mgns'][0]):
        for y in range(modules['mgns'][1]):
            modules['mgns'][8][x][y][0] = 0.

            
def intertrial_interval(modules, script_params):
    """
    resets the visual inputs and short-term memory using given parameters

    """

    # reset D1
    for x in range(modules['efd1'][0]):
        for y in range(modules['efd1'][1]):
            modules['efd1'][8][x][y][0] = script_params[2]
	    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['mgns'][0]):
        for y in range(modules['mgns'][1]):
            modules['mgns'][8][x][y][0] = script_params[2]

    # turn attention to 'LO', as the current trial has ended
    modules['atts'][8][0][0][0] = script_params[0]


# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
simulation_events = {        

    # REST BLOCK OF 4400 TIMESTEPS
    '0' : intertrial_interval,
    
    ####### FIRST BLOCK OF 4 DMS TRIALS (MATCH, MISMATCH, MISMATCH, MATCH)

    '100': s1_up_01,
    '110': s1_up_02,
    '120': s1_up_03,
    '130': s1_up_04,
    '140': s1_up_05,
    '150': s1_down_01,
    '160': s1_down_02,
    '170': s1_down_03,
    '180': s1_down_04,
    '190': s1_down_05,

    '200': delay_period,

    '400': s2_down_01,
    '410': s2_down_02,
    '420': s2_down_03,
    '430': s2_down_04,
    '440': s2_down_05,
    '450': s2_up_01,
    '460': s2_up_02,
    '470': s2_up_03,
    '480': s2_up_04,
    '490': s2_up_05,
    
    '500': intertrial_interval,

    '900': s1_up_01,
    '910': s1_up_02,
    '920': s1_up_03,
    '930': s1_up_04,
    '940': s1_up_05,
    '950': s1_down_01,
    '960': s1_down_02,
    '970': s1_down_03,
    '980': s1_down_04,
    '990': s1_down_05,
    
    '1000': delay_period,

    '1200': s1_up_01,
    '1210': s1_up_02,
    '1220': s1_up_03,
    '1230': s1_up_04,
    '1240': s1_up_05,
    '1250': s1_down_01,
    '1260': s1_down_02,
    '1270': s1_down_03,
    '1280': s1_down_04,
    '1290': s1_down_05,

    '1300': intertrial_interval,
    
    ################### END OF DMS BLOCK ####################################
    
}


##- EoF -##
