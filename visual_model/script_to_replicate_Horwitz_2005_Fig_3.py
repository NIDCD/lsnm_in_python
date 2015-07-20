#!/usr/bin/python
#
# The following script replicates the results of Horwitz, Warner et al (2005), Figure 3.
#
# There are 6 trials total: 3 DMS trials and 3 control trials
#
# Total number of timesteps is 6600 = 33 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# The DMS trials are MATCH, MISMATCH, MATCH. The attention parameter in DMS trials is 0.3
# The control trials are 'passive viewing': scrambled shapes are presented and low attention (0.05)
# is used.
#
# The first 200 timesteps = 1000 ms we do nothing. We assume 1 timestep = 5 ms, as in Horwitz
# et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
                
# now we present S1 by manually inserting it into the MGN module and leaving S1 there
# for 200 timesteps (1 second).

lo_att_level = 0.05
hi_att_level = 0.3
lo_inp_level = 0.05
hi_inp_level = 0.7

def delay_period(modules,
                 low_att_level, hi_att_level,
                 low_inp_level, hi_inp_level):
    
    "modifies neural network with delay period parameters given"
    
    modules['atts'][8][0][0][0] = hi_att_level
    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['lgns'][0]):
        for y in range(modules['lgns'][1]):
            modules['lgns'][8][x][y][0] = low_inp_level

def o_shape(modules,
            low_att_level, hi_att_level,
            low_inp_level, hi_inp_level):

    "gives an o-shaped visual input to neural network with parameters given"
    
    modules['atts'][8][0][0][0] = hi_att_level
    
    # insert the inputs stimulus into LGN and see what happens
    # the following stimulus is an 'O' shape
    modules['lgns'][8][4][3][0] = hi_inp_level
    modules['lgns'][8][4][4][0] = hi_inp_level
    modules['lgns'][8][4][5][0] = hi_inp_level
    modules['lgns'][8][4][6][0] = hi_inp_level
    modules['lgns'][8][4][7][0] = hi_inp_level
    modules['lgns'][8][4][8][0] = hi_inp_level
    modules['lgns'][8][8][3][0] = hi_inp_level
    modules['lgns'][8][8][4][0] = hi_inp_level
    modules['lgns'][8][8][5][0] = hi_inp_level
    modules['lgns'][8][8][6][0] = hi_inp_level
    modules['lgns'][8][8][7][0] = hi_inp_level
    modules['lgns'][8][8][8][0] = hi_inp_level
    modules['lgns'][8][5][3][0] = hi_inp_level
    modules['lgns'][8][6][3][0] = hi_inp_level
    modules['lgns'][8][7][3][0] = hi_inp_level
    modules['lgns'][8][5][8][0] = hi_inp_level
    modules['lgns'][8][6][8][0] = hi_inp_level
    modules['lgns'][8][7][8][0] = hi_inp_level
    
def t_shape(modules,
            low_att_level, hi_att_level,
            low_inp_level, hi_inp_level):
    
    modules['atts'][8][0][0][0] = hi_att_level

    # insert the inputs stimulus into LGN and see what happens
    # the following is an 'T' shape
    modules['lgns'][8][3][0][0] = hi_inp_level
    modules['lgns'][8][3][1][0] = hi_inp_level
    modules['lgns'][8][3][2][0] = hi_inp_level
    modules['lgns'][8][3][3][0] = hi_inp_level
    modules['lgns'][8][3][4][0] = hi_inp_level
    modules['lgns'][8][3][5][0] = hi_inp_level
    modules['lgns'][8][3][6][0] = hi_inp_level
    modules['lgns'][8][3][7][0] = hi_inp_level
    modules['lgns'][8][3][8][0] = hi_inp_level
    modules['lgns'][8][2][6][0] = hi_inp_level
    modules['lgns'][8][2][7][0] = hi_inp_level
    modules['lgns'][8][1][6][0] = hi_inp_level
    modules['lgns'][8][1][7][0] = hi_inp_level
    modules['lgns'][8][0][6][0] = hi_inp_level
    modules['lgns'][8][0][7][0] = hi_inp_level
    
def intertrial_interval(modules,
                        low_att_level, hi_att_level,
                        low_inp_level, hi_inp_level):
    # reset D1
    for x in range(modules['efd1'][0]):
        for y in range(modules['efd1'][1]):
            modules['efd1'][8][x][y][0] = 0.25
	    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['lgns'][0]):
        for y in range(modules['lgns'][1]):
            modules['lgns'][8][x][y][0] = low_inp_level

    # turn attention to 'LO', as the current trial has ended
    modules['atts'][8][0][0][0] = low_att_level
    
simulation_events = {        
            
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

}
