#!/usr/bin/python
#
# The following script replicates the results of Rong et al (2011), Tones during PSL
#
# Type of stimuli: Tones
# We have one condition: Passive Listening (PSL)
#
# We have a total of 12 trials, and they are all DMS trials
#
# The number of timesteps in each trial is 800 timesteps = 2.8 seconds
#
# For baseline setting purposes, we have included a 200 timestep "do-nothing" step, only at the beginning
# of the experiment
#
# The total number of timesteps is  12 x 800 = 9600 timesteps
#
# Trials are ordered as follows: MATCH, MISMATCH, MATCH, MISMATCH
#
# The attention parameter in all trials is constant at 0.01
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
#                
# We present stimuli to the auditory LSNM network by manually inserting it into the MGN module
# and leaving the stimuli there for 100 timesteps (350 ms).

# define the simulation time in total number of timesteps
LSNM_simulation_time = 9800

# Define list of parameters the script is going to need to modify the LSNM neural network
# They are organized in the following order:
# [lo_att_level, hi_att_level, lo_inp_level, hi_inp_level, att_step, ri1, ri2]
script_params = [0.01, 0.01, 0.0, 1.0, 0.0, [], []]

def tone_01(modules, script_params):
    
    """
    presents a tone to network using the given parameters
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is a tone
    modules['mgns'][8][0][41][0] = script_params[3]
    modules['mgns'][8][0][42][0] = script_params[3]

def tone_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """

    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][56][0] = script_params[3]


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

    # BASELINE BLOCK OF 200 TIMESTEPS
    '0' : intertrial_interval,
    
    ####### WHAT FOLLOWS IS 12 TRIALS USING Tones and ALTERNATING MATCH, MISMATCH, MATCH, MISMATCH,...

    # trial 1
    '200': tone_01,

    '300': delay_period,

    '500': tone_01,
    
    '600': intertrial_interval,

    # trial 2
    '1000': tone_01,
    
    '1100': delay_period,

    '1300': tone_02,

    '1400': intertrial_interval,

    # trial 3
    '1800': tone_02,

    '1900': delay_period,

    '2100': tone_02,
    
    '2200': intertrial_interval,

    # trial 4
    '2600': tone_02,
    
    '2700': delay_period,

    '2900': tone_01,

    '3000': intertrial_interval,

    # trial 5
    '3400': tone_01,

    '3500': delay_period,

    '3700': tone_01,
    
    '3800': intertrial_interval,

    # trial 6
    '4200': tone_01,
    
    '4300': delay_period,

    '4500': tone_02,

    '4600': intertrial_interval,

    # trial 7
    '5000': tone_02,

    '5100': delay_period,

    '5300': tone_02,
    
    '5400': intertrial_interval,

    # trial 8
    '5800': tone_02,
     
    '5900': delay_period,

    '6100': tone_01,

    '6200': intertrial_interval,

    # trial 9
    '6600': tone_01,

    '6700': delay_period,

    '6900': tone_01,
    
    '7000': intertrial_interval,

    # trial 10
    '7400': tone_01,
    
    '7500': delay_period,

    '7700': tone_02,

    '7800': intertrial_interval,

    # trial 11
    '8200': tone_02,

    '8300': delay_period,

    '8500': tone_02,
    
    '8600': intertrial_interval,

    # trial 12
    '9000': tone_02,
    
    '9100': delay_period,

    '9300': tone_01,

    '9400': intertrial_interval,
    
    ################### END OF DMS BLOCK ####################################
    
}


##- EoF -##
