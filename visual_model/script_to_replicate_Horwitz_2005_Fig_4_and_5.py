#!/usr/bin/python
#
# The following script replicates the results of Horwitz, Warner et al (2005), Figure 4.
#
# There are 3 DMS trials and 3 control trials for each of six levels of attention
#
# The total number of timesteps is 39600 = 198 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# Each block is 3300 timesteps = 16.5 seconds
#
# Both DMS trials and control trials are: MATCH, MISMATCH, MATCH.
# The attention parameter in DMS trials is varied from 0.2 to 0.3 in steps of 0.02
# The control trials constitute 'passive viewing': degraded shapes are presented and
# the same low attention (0.05) is used throughout the control trials
#
# The first 200 timesteps = 1000 ms we do nothing. We assume 1 timestep = 5 ms, as in
# Horwitz et al (2005)
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
script_params = [0.05, 0.2, 0.05, 0.7, 0.02, [], []]

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

def increase_attention(modules, script_params):
    """
    Increases 'hi_att_level' by a step given by 'att_step'

    """

    script_params[1] = script_params[1] + script_params[4]

    
# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
simulation_events = {        
    ################### FIRST BLOCK OF 3 DMS TRIALS (MATCH, MISMATCH, MATCH)
    '200': o_shape,                

    '400': delay_period,

    '700': o_shape,

    '900': intertrial_interval,             

    '1300': o_shape,

    '1500': delay_period,

    '1800': t_shape,

    '2000': intertrial_interval,
             
    '2400': t_shape,

    '2600': delay_period,

    '2900': t_shape,

    '3100': intertrial_interval,

    ################### FIRST BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
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

    '6400': intertrial_interval,

    ################### SECOND BLOCK OF 3 DMS TRIALS (MATCH, MISMATCH, MATCH)
    '6799': increase_attention,
    '6800': o_shape,                

    '7000': delay_period,

    '7300': o_shape,

    '7500': intertrial_interval,             

    '7900': o_shape,

    '8100': delay_period,

    '8400': t_shape,

    '8600': intertrial_interval,

    '9000': t_shape,

    '9200': delay_period,

    '9500': t_shape,

    '9700': intertrial_interval,

    ################### SECOND BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
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

    '13000': intertrial_interval,
    
    ################### THIRD BLOCK OF 3 DMS TRIALS (MATCH, MISMATCH, MATCH)
    '13399': increase_attention,
    '13400': o_shape,                

    '13600': delay_period,

    '13900': o_shape,

    '14100': intertrial_interval,             

    '14500': o_shape,

    '14700': delay_period,

    '15000': t_shape,

    '15200': intertrial_interval,
             
    '15600': t_shape,

    '15800': delay_period,

    '16100': t_shape,

    '16300': intertrial_interval,

    ################### THIRD BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
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

    '19600': intertrial_interval,
    
    ################### FOURTH BLOCK OF 3 DMS TRIALS (MATCH, MISMATCH, MATCH)
    '19999': increase_attention,
    '20000': o_shape,                

    '20200': delay_period,

    '20500': o_shape,

    '20700': intertrial_interval,             

    '21100': o_shape,

    '21300': delay_period,

    '21600': t_shape,

    '21800': intertrial_interval,
             
    '22200': t_shape,

    '22400': delay_period,

    '22700': t_shape,

    '22900': intertrial_interval,

    ################### FOURTH BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
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

    '26200': intertrial_interval,
    
    ################### FIFTH BLOCK OF 3 DMS TRIALS (MATCH, MISMATCH, MATCH)
    '26599': increase_attention,
    '26600': o_shape,                

    '26800': delay_period,

    '27100': o_shape,

    '27300': intertrial_interval,             

    '27700': o_shape,

    '27900': delay_period,

    '28200': t_shape,

    '28400': intertrial_interval,
             
    '28800': t_shape,

    '29000': delay_period,

    '29300': t_shape,

    '29500': intertrial_interval,

    ################### FIFTH BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
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

    '32800': intertrial_interval,
    
    ################### SIXTH BLOCK OF 3 DMS TRIALS (MATCH, MISMATCH, MATCH)
    '33199': increase_attention,
    '33200': o_shape,                

    '33400': delay_period,

    '33700': o_shape,

    '33900': intertrial_interval,             

    '34300': o_shape,

    '34500': delay_period,

    '34800': t_shape,

    '35000': intertrial_interval,
             
    '35400': t_shape,

    '35600': delay_period,

    '35900': t_shape,

    '36100': intertrial_interval,

    ################### SIXTH BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
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
