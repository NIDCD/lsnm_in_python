#!/usr/bin/python
#
# Script created on December 5 2016
#
# There are 6 Delayed match-to-sample (DMS) task blocks in this script, interspersed
# with 6 rest blocks
#
# The total number of timesteps is 39600 = 198 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# Each block is 3300 timesteps = 16.5 seconds
#
# Each task block is composed of 3 DMS trials in the following order: MATCH, MISMATCH, MATCH.
# The attention parameter in the task trials is 0.3 and the attention parameter in the rest
# blocks is 0.05
#
# We assume 1 timestep = 5 ms, as in Horwitz et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
#                
# We present stimuli to the visual LSNM network by manually inserting it into the MGN module
# and leaving the stimuli there for 200 timesteps (1 second).

# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = 39600

# Define list of parameters the script is going to need to modify the LSNM neural network
# They are organized in the following order:
# [lo_att_level, hi_att_level, lo_inp_level, hi_inp_level, att_step, ri1, ri2]
script_params = [0.05, 0.3, 0.05, 0.7, 0.02, [], []]

def o_shape(modules, script_params):
    
    """
    generates an o-shaped visual input to neural network with parameters given
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into LGN and see what happens
    # the following stimulus is an 'O' shape
    modules['lgns'][8][4][3][0] = script_params[3]
    modules['lgns'][8][4][4][0] = script_params[3]
    modules['lgns'][8][4][5][0] = script_params[3]
    modules['lgns'][8][4][6][0] = script_params[3]
    modules['lgns'][8][4][7][0] = script_params[3]
    modules['lgns'][8][4][8][0] = script_params[3]
    modules['lgns'][8][8][3][0] = script_params[3]
    modules['lgns'][8][8][4][0] = script_params[3]
    modules['lgns'][8][8][5][0] = script_params[3]
    modules['lgns'][8][8][6][0] = script_params[3]
    modules['lgns'][8][8][7][0] = script_params[3]
    modules['lgns'][8][8][8][0] = script_params[3]
    modules['lgns'][8][5][3][0] = script_params[3]
    modules['lgns'][8][6][3][0] = script_params[3]
    modules['lgns'][8][7][3][0] = script_params[3]
    modules['lgns'][8][5][8][0] = script_params[3]
    modules['lgns'][8][6][8][0] = script_params[3]
    modules['lgns'][8][7][8][0] = script_params[3]
    
def t_shape(modules, script_params):
    
    """
    generates a t-shaped visual input to neural network with parameters given"
    
    """
    modules['atts'][8][0][0][0] = script_params[1]

    # insert the inputs stimulus into LGN and see what happens
    # the following is a 'T' shape
    modules['lgns'][8][3][0][0] = script_params[3]
    modules['lgns'][8][3][1][0] = script_params[3]
    modules['lgns'][8][3][2][0] = script_params[3]
    modules['lgns'][8][3][3][0] = script_params[3]
    modules['lgns'][8][3][4][0] = script_params[3]
    modules['lgns'][8][3][5][0] = script_params[3]
    modules['lgns'][8][3][6][0] = script_params[3]
    modules['lgns'][8][3][7][0] = script_params[3]
    modules['lgns'][8][0][6][0] = script_params[3]
    modules['lgns'][8][1][6][0] = script_params[3]
    modules['lgns'][8][1][7][0] = script_params[3]
    modules['lgns'][8][2][6][0] = script_params[3]
    modules['lgns'][8][2][7][0] = script_params[3]
    modules['lgns'][8][4][6][0] = script_params[3]
    modules['lgns'][8][4][7][0] = script_params[3]
    modules['lgns'][8][5][6][0] = script_params[3]
    modules['lgns'][8][5][7][0] = script_params[3]
    modules['lgns'][8][6][6][0] = script_params[3]

def delay_period(modules, script_params):
    
    """
    modifies neural network with delay period parameters given

    """
    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['lgns'][0]):
        for y in range(modules['lgns'][1]):
            modules['lgns'][8][x][y][0] = script_params[2]

def intertrial_interval(modules, script_params):

    """
    resets the visual inputs and short-term memory using given parameters

    """

    # reset D1
    for x in range(modules['efd1'][0]):
        for y in range(modules['efd1'][1]):
            modules['efd1'][8][x][y][0] = script_params[2]
	    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['lgns'][0]):
        for y in range(modules['lgns'][1]):
            modules['lgns'][8][x][y][0] = script_params[2]

    # turn attention to 'LO', as the current trial has ended
    modules['atts'][8][0][0][0] = script_params[0]
    
# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
simulation_events = {
    '0'   : intertrial_interval,             # rest block begins
    ################### BLOCK 1
    '3500': o_shape,

    '3700': delay_period,

    '4000': o_shape,

    '4200': intertrial_interval,

    '4600': o_shape,

    '4800': delay_period,

    '5100': t_shape,

    '5300': intertrial_interval,

    '5700': t_shape,

    '5900': delay_period,

    '6200': t_shape,

    '6400': intertrial_interval,             # rest block begins

    ################### BLOCK 2
    '10100': o_shape,

    '10300': delay_period,

    '10600': o_shape,

    '10800': intertrial_interval,

    '11200': o_shape,

    '11400': delay_period,

    '11700': t_shape,

    '11900': intertrial_interval,

    '12300': t_shape,

    '12500': delay_period,

    '12800': t_shape,

    '13000': intertrial_interval,             # rest block begins
    
    ################### BLOCK 3
    '16700': o_shape,

    '16900': delay_period,

    '17200': o_shape,

    '17400': intertrial_interval,

    '17800': o_shape,

    '18000': delay_period,

    '18300': t_shape,

    '18500': intertrial_interval,

    '18900': t_shape,

    '19100': delay_period,

    '19400': t_shape,

    '19600': intertrial_interval,             # rest block begins
    
    ################### BLOCK 4
    '23300': o_shape,

    '23500': delay_period,

    '23800': o_shape,

    '24000': intertrial_interval,

    '24400': o_shape,

    '24600': delay_period,

    '24900': t_shape,

    '25100': intertrial_interval,

    '25500': t_shape,

    '25700': delay_period,

    '26000': t_shape,

    '26200': intertrial_interval,             # rest block begins
    
    ################### BLOCK 5
    '29900': o_shape,

    '30100': delay_period,

    '30400': o_shape,

    '30600': intertrial_interval,

    '31000': o_shape,

    '31200': delay_period,

    '31500': t_shape,

    '31700': intertrial_interval,

    '32100': t_shape,

    '32300': delay_period,

    '32600': t_shape,

    '32800': intertrial_interval,             # rest block begins
    
    ################### BLOCK 6
    '36500': o_shape,

    '36700': delay_period,

    '37000': o_shape,

    '37200': intertrial_interval,

    '37600': o_shape,

    '37800': delay_period,

    '38100': t_shape,

    '38300': intertrial_interval,

    '38700': t_shape,

    '38900': delay_period,

    '39200': t_shape,

    '39400': intertrial_interval,      

}


##- EoF -##
