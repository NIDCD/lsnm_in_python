#!/usr/bin/python
#
# Script created on December 5 2016
#
# There are 3 Passive Viewing trials and 3 control trials (rest) for each of six blocks
#
# The total number of timesteps is 39600 = 198 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# Each block is 3300 timesteps = 16.5 seconds
#
# The attention parameter in all trials (passive viewing and rest) is 0.05
# 'passive viewing': degraded shapes are presented and
# low attention (0.05) is used throughout
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

# the following is random shape1, this shape has the same luminance as an 'O'
# Contributed by John Gilbert
rand_shape1 = rdm.sample(range(81),18)
rand_indeces1 = np.unravel_index(rand_shape1,(9,9))
script_params[5] = zip(*rand_indeces1)

# A second random shape is inserted for a mismatch
# Contributed by John Gilbert
rand_shape2 = rdm.sample(range(81),18)
rand_indeces2 = np.unravel_index(rand_shape2,(9,9))
script_params[6] = zip(*rand_indeces2)
        
def random_shape_1(modules, script_params):
    """
    generates a random visual input to neural network with parameters given
    
    """
    modules['atts'][8][0][0][0] = script_params[0]

    for k1 in range(len(script_params[5])):
        modules['lgns'][8][script_params[5][k1][0]][script_params[5][k1][1]][0] = script_params[3]
    
def random_shape_2(modules, script_params):
    """
    generates a random visual input to neural network with parameters given
    
    """
    
    modules['atts'][8][0][0][0] = script_params[0]

    for k1 in range(len(script_params[6])):
        modules['lgns'][8][script_params[6][k1][0]][script_params[6][k1][1]][0] = script_params[3]

    
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
    '0'  : intertrial_interval,             # rest block begins
    ################### FIRST BLOCK OF PASSIVE VIEWING TRIALS (MATCH, MISMATCH, MATCH)
    '3500': random_shape_1,

    '3700': delay_period,

    '4000': random_shape_1,

    '4200': intertrial_interval,

    '4600': random_shape_1,

    '4800': delay_period,

    '5100': random_shape_2,

    '5300': intertrial_interval,

    '5700': random_shape_2,

    '5900': delay_period,

    '6200': random_shape_2,

    '6400': intertrial_interval,             # rest block begins

    ################### SECOND BLOCK OF PASSIVE VIEWING TRIALS (MATCH, MISMATCH, MATCH)
    '10100': random_shape_1,

    '10300': delay_period,

    '10600': random_shape_1,

    '10800': intertrial_interval,

    '11200': random_shape_1,

    '11400': delay_period,

    '11700': random_shape_2,

    '11900': intertrial_interval,

    '12300': random_shape_2,

    '12500': delay_period,

    '12800': random_shape_2,

    '13000': intertrial_interval,             # rest block begins
    
    ################### THIRD BLOCK OF PASSIVE VIEWING TRIALS (MATCH, MISMATCH, MATCH)
    '16700': random_shape_1,

    '16900': delay_period,

    '17200': random_shape_1,

    '17400': intertrial_interval,

    '17800': random_shape_1,

    '18000': delay_period,

    '18300': random_shape_2,

    '18500': intertrial_interval,

    '18900': random_shape_2,

    '19100': delay_period,

    '19400': random_shape_2,

    '19600': intertrial_interval,             # rest block begins
    
    ################### FOURTH BLOCK OF PASSIVE VIEWING TRIALS (MATCH, MISMATCH, MATCH)
    '23300': random_shape_1,

    '23500': delay_period,

    '23800': random_shape_1,

    '24000': intertrial_interval,

    '24400': random_shape_1,

    '24600': delay_period,

    '24900': random_shape_2,

    '25100': intertrial_interval,

    '25500': random_shape_2,

    '25700': delay_period,

    '26000': random_shape_2,

    '26200': intertrial_interval,             # rest block begins
    
    ################### FIFTH BLOCK OF PASSIVE VIEWING TRIALS (MATCH, MISMATCH, MATCH)
    '29900': random_shape_1,

    '30100': delay_period,

    '30400': random_shape_1,

    '30600': intertrial_interval,

    '31000': random_shape_1,

    '31200': delay_period,

    '31500': random_shape_2,

    '31700': intertrial_interval,

    '32100': random_shape_2,

    '32300': delay_period,

    '32600': random_shape_2,

    '32800': intertrial_interval,             # rest block begins
    
    ################### SIXTH BLOCK OF PASSIVE VIEWING TRIALS (MATCH, MISMATCH, MATCH)
    '36500': random_shape_1,

    '36700': delay_period,

    '37000': random_shape_1,

    '37200': intertrial_interval,

    '37600': random_shape_1,

    '37800': delay_period,

    '38100': random_shape_2,

    '38300': intertrial_interval,

    '38700': random_shape_2,

    '38900': delay_period,

    '39200': random_shape_2,

    '39400': intertrial_interval,

}


##- EoF -##
