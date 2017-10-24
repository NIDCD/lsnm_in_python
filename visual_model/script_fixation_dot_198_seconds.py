#!/usr/bin/python
#
# Script created on October 4 2017
#
# Fixation script (AKA 'Resting State' in the neuroimaging literature)
#
# The total number of timesteps is 39600 = 198 seconds
#
# We assume 1 timestep = 5 ms, as in Horwitz et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
#                
# We present stimuli to the visual LSNM network by manually inserting it into the LGN module

# define the simulation time in total number of timesteps
# Each timestep is roughly equivalent to 5ms
LSNM_simulation_time = 39600

# Define list of parameters the script is going to need to modify the LSNM neural network
# They are organized in the following order:
# [lo_att_level, hi_att_level, lo_inp_level, hi_inp_level, att_step, ri1, ri2]
script_params = [0.05, 0.3, 0.05, 0.2, 0.02, [], []]

def dot_shape(modules, script_params):
    
    """
    generates a cross-shaped visual input to neural network with parameters given
    
    """
    
    modules['atts'][8][0][0][0] = script_params[0]
    
    # insert the inputs stimulus into LGN and see what happens
    # the following stimulus is an 'cross' shape
    modules['lgns'][8][4][4][0] = script_params[3]
    
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
    '0'   : intertrial_interval,             
    '3500': dot_shape,
    '39400': intertrial_interval      

}


##- EoF -##
