#!/usr/bin/python
#
# The following script tests a DMS task with forgetting during a long ITI
#
# There is 1 trial total with a long ITI
#
# Total number of timesteps is 6600 = 33 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# The first 200 timesteps = 1000 ms we do nothing. We assume 1 timestep = 5 ms, as in Horwitz
# et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
                
# now we present S1 by manually inserting it into the MGN module and leaving S1 there
# for 200 timesteps (1 second).

# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = 2200
                
# Define list of parameters the the script is going to need to modify the LSNM neural network
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
    turns off input stimulus and lowers attention level

    """

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
            
    '200': o_shape,                

    '400': delay_period,

    '700': o_shape,

    '900': intertrial_interval,             

    '1300': t_shape,

    '1500': delay_period,

    '1800': o_shape,

    '2000': intertrial_interval,

}


##- EoF -##
