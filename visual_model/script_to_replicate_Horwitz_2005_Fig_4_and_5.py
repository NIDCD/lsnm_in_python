#!/usr/bin/python
#
# The following script replicates the results of Horwitz, Warner et al (2005), Figures 4 and 5.
#
# There are 36 trials total, divided in groups of 6 trials: each group has a level of attention
# that increases from 0.2 to 0.3 in steps of 0.02.
#
# Within each attention level, we have 3 DMS trials and 3 control trials
#
# The DMS trials are MATCH, MISMATCH, MATCH. The attention parameter in DMS trials is 0.3
# The control trials are 'passive viewing': random shapes are presented and low attention (0.05)
# is used. Passive viewing trials are also organized as MATCH, MISMATCH, MATCH.
#
# Total number of timesteps is 39600 = 198 seconds
#
# The number of timesteps in each trial is 1100 = 5.5 seconds
#
# The first 200 timesteps = 1000 ms we do nothing. We assume 1 timestep = 5 ms, as in Horwitz
# et al (2005)
#
# To maintain consistency with Husain et al (2004) and Tagamets and Horwitz (1998),
# we are assuming that each simulation timestep is equivalent to 5 milliseconds
# of real time. 
                
lo_att_level = 0.05
hi_att_level = 0.3
lo_inp_level = 0.05
md_inp_level = 0.54
hi_inp_level = 0.7

# the following is random shape1, this shape has the same luminance as an 'O'
rand_shape1 = rdm.sample(range(81),18)
rand_indeces1 = np.unravel_index(rand_shape1,(9,9))
ri1 = zip(*rand_indeces1)

# A second random shape in inserted for a mismatch
rand_shape2 = rdm.sample(range(81),18)
rand_indeces2 = np.unravel_index(rand_shape2,(9,9))
ri2 = zip(*rand_indeces2)
        
def o_shape(modules,
            low_att_level, hi_att_level,
            low_inp_level, md_inp_level, hi_inp_level):
    """
    generates an o-shaped visual input to neural network with parameters given"
    
    """
    
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
            low_inp_level, md_inp_level, hi_inp_level):
    
    """
    generates a t-shaped visual input to neural network with parameters given"
    
    """
    modules['atts'][8][0][0][0] = hi_att_level

    # insert the inputs stimulus into LGN and see what happens
    # the following is a 'T' shape
    modules['lgns'][8][3][0][0] = 0.7
    modules['lgns'][8][3][1][0] = 0.7
    modules['lgns'][8][3][2][0] = 0.7
    modules['lgns'][8][3][3][0] = 0.7
    modules['lgns'][8][3][4][0] = 0.7
    modules['lgns'][8][3][5][0] = 0.7
    modules['lgns'][8][3][6][0] = 0.7
    modules['lgns'][8][3][7][0] = 0.7
    modules['lgns'][8][0][6][0] = 0.7
    modules['lgns'][8][1][6][0] = 0.7
    modules['lgns'][8][1][7][0] = 0.7
    modules['lgns'][8][2][6][0] = 0.7
    modules['lgns'][8][2][7][0] = 0.7
    modules['lgns'][8][4][6][0] = 0.7
    modules['lgns'][8][4][7][0] = 0.7
    modules['lgns'][8][5][6][0] = 0.7
    modules['lgns'][8][5][7][0] = 0.7
    modules['lgns'][8][6][6][0] = 0.7

def random_shape_1(modules,
                    low_att_level, hi_att_level,
                    low_inp_level, md_inp_level, hi_inp_level):
    """
    generates a random visual input to neural network with parameters given
    
    """
    for k1 in range(len(ri1)):
        modules['lgns'][8][ri1[k1][0]][ri1[k1][1]][0] = md_inp_level
    
def random_shape_2(modules,
                    low_att_level, hi_att_level,
                    low_inp_level, md_inp_level, hi_inp_level):
    """
    generates a random visual input to neural network with parameters given
    
    """
    
    for k1 in range(len(ri2)):
        modules['lgns'][8][ri2[k1][0]][ri2[k1][1]][0] = md_inp_level

    
def delay_period(modules,
                 low_att_level, hi_att_level,
                 low_inp_level, md_inp_level, hi_inp_level):
    
    """
    modifies neural network with delay period parameters given

    """
    
    modules['atts'][8][0][0][0] = hi_att_level
    
    # turn off input stimulus but leave small level of activity there
    for x in range(modules['lgns'][0]):
        for y in range(modules['lgns'][1]):
            modules['lgns'][8][x][y][0] = low_inp_level

def intertrial_interval(modules,
                        low_att_level, hi_att_level,
                        low_inp_level, md_inp_level, hi_inp_level):
    """
    resets the visual inputs and short-term memory using given parameters

    """

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

    
# define a dictionary of simulation events functions, each associated with
# a specific simulation timestep
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

}


##- EoF -##
