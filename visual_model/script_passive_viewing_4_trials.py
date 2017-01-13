#!/usr/bin/python
#
# Script created on January 12 2017
#
# There is a rest block followed by 3 Passive Viewing trials
#
# The total number of timesteps is 1400 = 7 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# Each block is 3300 timesteps = 16.5 seconds
#
# The attention parameter in all trials (passive viewing and rest) is 0.05
# 'passive viewing'
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
LSNM_simulation_time = 1400

# Define list of parameters the script is going to need to modify the LSNM neural network
# They are organized in the following order:
# [lo_att_level, hi_att_level, lo_inp_level, hi_inp_level, att_step, ri1, ri2]
script_params = [0.05, 0.3, 0.05, 0.7, 0.02, [], []]

def T_shape_LA(modules, script_params): 
     
    """ 
    generates a t-shaped visual input to neural network with parameters given" 
    Good intensity is about .69625 and this appears to be very sensitive 
    """ 
    modules['atts'][8][0][0][0] = script_params[0] 
 
    # insert the inputs stimulus into LGN and see what happens 
    # the following is a 'T' shape 
    modules['lgns'][8][0][0][0] = 0.7 #script_params[3]
    modules['lgns'][8][0][1][0] = 0.7 #script_params[3] 
    modules['lgns'][8][0][2][0] = 0.7 #script_params[3] 
    modules['lgns'][8][0][3][0] = 0.7 #script_params[3] 
    modules['lgns'][8][0][4][0] = 0.7 #script_params[3] 
    modules['lgns'][8][0][5][0] = 0.7 #script_params[3] 
    modules['lgns'][8][0][6][0] = 0.7 #script_params[3]
    modules['lgns'][8][0][7][0] = 0.7 #script_params[3]
    modules['lgns'][8][0][8][0] = 0.7 #script_params[3]
    # stem
    modules['lgns'][8][1][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][2][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][3][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][4][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][5][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][6][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][7][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][8][4][0] = 0.7 #script_params[3]
 
def cross_shape_LA(modules, script_params): 
    """ 
    generates a cross(+)-shaped visual input to neural network with parameters given"
    Good intensity is about .69625 and this appears to be very sensitive 
    """ 
    modules['atts'][8][0][0][0] = script_params[0] 
 
    # insert the inputs stimulus into LGN and see what happens 
    # the following is a 'cross' shape
    modules['lgns'][8][4][0][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][1][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][2][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][3][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][4][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][5][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][6][0] = 0.7 #script_params[3] 
    modules['lgns'][8][4][7][0] = 0.7 #script_params[3]
    modules['lgns'][8][4][8][0] = 0.7 #script_params[3]
    # the other part of the t
    modules['lgns'][8][0][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][1][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][2][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][3][4][0] = 0.7 #script_params[3] 
    modules['lgns'][8][5][4][0] = 0.7 #script_params[3] 
    modules['lgns'][8][6][4][0] = 0.7 #script_params[3]
    modules['lgns'][8][7][4][0] = 0.7 #script_params[3] 
    modules['lgns'][8][8][4][0] = 0.7 #script_params[3]        
    
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
    '500': T_shape_LA,

    '700': delay_period,

    '1000': cross_shape_LA,

    '1200': intertrial_interval,

}


##- EoF -##
