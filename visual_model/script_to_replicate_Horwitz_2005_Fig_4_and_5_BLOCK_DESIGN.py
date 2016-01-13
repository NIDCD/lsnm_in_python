#!/usr/bin/python
#
# The following script replicates the results of Horwitz, Warner et al (2005), Figure 4.
#
# The script has the following 5 blocks: REST, DMS, REST, CTL, REST
#
# We have a total of 8 trials (4 DMS and 4 CTL) in each block
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
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
# We present stimuli to the visual LSNM network by manually inserting it into the MGN module
# and leaving the stimuli there for 200 timesteps (1 second).

# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = 22000

# Define list of parameters the script is going to need to modify the LSNM neural network
# They are organized in the following order:
# [lo_att_level, hi_att_level, lo_inp_level, hi_inp_level, att_step, ri1, ri2]
script_params = [0.05, 0.3, 0.05, 0.7, 0.0, [], []]

# the following is random shape1,
# Contributed by John Gilbert
rand_shape1 = rdm.sample(range(81),5)
rand_indeces1 = np.unravel_index(rand_shape1,(9,9))
script_params[5] = zip(*rand_indeces1)

# A second random shape in inserted for a mismatch
# Contributed by John Gilbert
rand_shape2 = rdm.sample(range(81),5)
rand_indeces2 = np.unravel_index(rand_shape2,(9,9))
script_params[6] = zip(*rand_indeces2)
        
def v_bar(modules, script_params):
    
    """
    generates an vertical bar visual input to neural network with parameters given
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into LGN and see what happens
    # the following stimulus is a vertical bar
    modules['lgns'][8][4][2][0] = script_params[3]
    modules['lgns'][8][4][3][0] = script_params[3]
    modules['lgns'][8][4][4][0] = script_params[3]
    modules['lgns'][8][4][5][0] = script_params[3]
    modules['lgns'][8][4][6][0] = script_params[3]

def h_bar(modules, script_params):
    
    """
    generates a horizontal bar visual input to neural network with parameters given"
    
    """
    modules['atts'][8][0][0][0] = script_params[1]

    # insert the inputs stimulus into LGN and see what happens
    # the following is a horizontal bar
    modules['lgns'][8][2][4][0] = script_params[3]
    modules['lgns'][8][3][4][0] = script_params[3]
    modules['lgns'][8][4][4][0] = script_params[3]
    modules['lgns'][8][5][4][0] = script_params[3]
    modules['lgns'][8][6][4][0] = script_params[3]

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

    # REST BLOCK OF 4400 TIMESTEPS
    
    ####### FIRST BLOCK OF 4 DMS TRIALS (MATCH, MISMATCH, MISMATCH, MATCH)
    '4400': v_bar,                

    '4600': delay_period,

    '4900': v_bar,

    '5100': intertrial_interval,             

    '5500': v_bar,

    '5700': delay_period,

    '6000': h_bar,

    '6200': intertrial_interval,
             
    '6600': h_bar,

    '6800': delay_period,

    '7100': v_bar,

    '7300': intertrial_interval,

    '7700': h_bar,

    '7900': delay_period,

    '8200': h_bar,

    '8400': intertrial_interval,

    ################### END OF DMS BLOCK ####################################
    
    # REST BLOCK OF 4400 TIMESTEPS

    ################### FIRST BLOCK OF 3 CONTROL TRIALS (MATCH, MISMATCH, MATCH)
    '13200': random_shape_1,

    '13400': delay_period,

    '13700': random_shape_1,

    '13900': intertrial_interval,

    '14300': random_shape_1,

    '14500': delay_period,

    '14800': random_shape_2,

    '15000': intertrial_interval,

    '15400': random_shape_2,

    '15600': delay_period,

    '15900': random_shape_1,

    '16100': intertrial_interval,

    '16500': random_shape_2,

    '16700': delay_period,

    '17000': random_shape_2,

    '17200': intertrial_interval,

    ################### END OF CONTROL BLOCK ####################################
    
    # REST BLOCK OF 4400 TIMESTEPS
    # final timepoint is 22000

}


##- EoF -##
