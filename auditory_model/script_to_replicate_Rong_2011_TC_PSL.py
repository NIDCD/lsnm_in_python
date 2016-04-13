#!/usr/bin/python
#
# The following script replicates the results of Rong et al (2011), Tonal Countours during PSL
#
# Type of stimuli: Tonal Contours (TC)
# We have one condition: Passive Listening (PSL)
#
# We have a total of 12 trials, and they are all PSL trials
#
# The number of timesteps in each trial is 800 timesteps = 2.8 seconds
#
# For baseline setting purposes, we have included a 200 timestep "do-nothing" step, only at the beginning
# of the experiment
#
# The total number of timesteps is  12 x 800 = 9600 timesteps (note that we add 200 extra tsteps at beginning)
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

def s1_up_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][42][0] = script_params[3]
    modules['mgns'][8][0][43][0] = script_params[3]

def s1_up_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """

    # reset previous activation
    modules['mgns'][8][0][42][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][43][0] = script_params[3]
    modules['mgns'][8][0][44][0] = script_params[3]

def s1_up_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][43][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][44][0] = script_params[3]
    modules['mgns'][8][0][45][0] = script_params[3]

def s1_up_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][44][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][45][0] = script_params[3]
    modules['mgns'][8][0][46][0] = script_params[3]

def s1_up_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][45][0] = script_params[3]
    modules['mgns'][8][0][46][0] = script_params[3]

def s1_down_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][46][0] = script_params[3]
    modules['mgns'][8][0][45][0] = script_params[3]

def s1_down_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][46][0] = script_params[3]
    modules['mgns'][8][0][45][0] = script_params[3]

def s1_down_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][46][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][45][0] = script_params[3]
    modules['mgns'][8][0][44][0] = script_params[3]

def s1_down_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][45][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][44][0] = script_params[3]
    modules['mgns'][8][0][43][0] = script_params[3]

def s1_down_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][44][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][43][0] = script_params[3]
    modules['mgns'][8][0][42][0] = script_params[3]
    
def s2_down_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    modules['atts'][8][0][0][0] = script_params[1]
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][58][0] = script_params[3]
    modules['mgns'][8][0][57][0] = script_params[3]

def s2_down_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][58][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][57][0] = script_params[3]
    modules['mgns'][8][0][56][0] = script_params[3]

def s2_down_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][57][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][56][0] = script_params[3]
    modules['mgns'][8][0][55][0] = script_params[3]

def s2_down_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][56][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][54][0] = script_params[3]

def s2_down_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][54][0] = script_params[3]
    
def s2_up_01(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][54][0] = script_params[3]
    modules['mgns'][8][0][55][0] = script_params[3]

def s2_up_02(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][54][0] = script_params[3]

def s2_up_03(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][54][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][55][0] = script_params[3]
    modules['mgns'][8][0][56][0] = script_params[3]

def s2_up_04(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][55][0] = 0.
    
    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][56][0] = script_params[3]
    modules['mgns'][8][0][57][0] = script_params[3]

def s2_up_05(modules, script_params):
    
    """
    generates an up sweep to neural network using the given parameters
    
    """
    
    # reset previous activation
    modules['mgns'][8][0][56][0] = 0.

    # insert the inputs stimulus into MGN and see what happens
    # the following stimulus is an up sweep
    modules['mgns'][8][0][57][0] = script_params[3]
    modules['mgns'][8][0][58][0] = script_params[3]

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
    
    ####### WHAT FOLLOWS IS 12 TRIALS USING TCs and ALTERNATING MATCH, MISMATCH, MATCH, MISMATCH,...

    # trial 1
    '200': s1_up_01,
    '210': s1_up_02,
    '220': s1_up_03,
    '230': s1_up_04,
    '240': s1_up_05,
    '250': s1_down_01,
    '260': s1_down_02,
    '270': s1_down_03,
    '280': s1_down_04,
    '290': s1_down_05,

    '300': delay_period,

    '500': s1_up_01,
    '510': s1_up_02,
    '520': s1_up_03,
    '530': s1_up_04,
    '540': s1_up_05,
    '550': s1_down_01,
    '560': s1_down_02,
    '570': s1_down_03,
    '580': s1_down_04,
    '590': s1_down_05,
    
    '600': intertrial_interval,

    # trial 2
    '1000': s1_up_01,
    '1010': s1_up_02,
    '1020': s1_up_03,
    '1030': s1_up_04,
    '1040': s1_up_05,
    '1050': s1_down_01,
    '1060': s1_down_02,
    '1070': s1_down_03,
    '1080': s1_down_04,
    '1090': s1_down_05,
    
    '1100': delay_period,

    '1300': s2_down_01,
    '1310': s2_down_02,
    '1320': s2_down_03,
    '1330': s2_down_04,
    '1340': s2_down_05,
    '1350': s2_up_01,
    '1360': s2_up_02,
    '1370': s2_up_03,
    '1380': s2_up_04,
    '1390': s2_up_05,

    '1400': intertrial_interval,

    # trial 3
    '1800': s1_up_01,
    '1810': s1_up_02,
    '1820': s1_up_03,
    '1830': s1_up_04,
    '1840': s1_up_05,
    '1850': s1_down_01,
    '1860': s1_down_02,
    '1870': s1_down_03,
    '1880': s1_down_04,
    '1890': s1_down_05,

    '1900': delay_period,

    '2100': s1_up_01,
    '2110': s1_up_02,
    '2120': s1_up_03,
    '2130': s1_up_04,
    '2140': s1_up_05,
    '2150': s1_down_01,
    '2160': s1_down_02,
    '2170': s1_down_03,
    '2180': s1_down_04,
    '2190': s1_down_05,
    
    '2200': intertrial_interval,

    # trial 4
    '2600': s1_up_01,
    '2610': s1_up_02,
    '2620': s1_up_03,
    '2630': s1_up_04,
    '2640': s1_up_05,
    '2650': s1_down_01,
    '2660': s1_down_02,
    '2670': s1_down_03,
    '2680': s1_down_04,
    '2690': s1_down_05,
    
    '2700': delay_period,

    '2900': s2_down_01,
    '2910': s2_down_02,
    '2920': s2_down_03,
    '2930': s2_down_04,
    '2940': s2_down_05,
    '2950': s2_up_01,
    '2960': s2_up_02,
    '2970': s2_up_03,
    '2980': s2_up_04,
    '2990': s2_up_05,

    '3000': intertrial_interval,

    # trial 5
    '3400': s1_up_01,
    '3410': s1_up_02,
    '3420': s1_up_03,
    '3430': s1_up_04,
    '3440': s1_up_05,
    '3450': s1_down_01,
    '3460': s1_down_02,
    '3470': s1_down_03,
    '3480': s1_down_04,
    '3490': s1_down_05,

    '3500': delay_period,

    '3700': s1_up_01,
    '3710': s1_up_02,
    '3720': s1_up_03,
    '3730': s1_up_04,
    '3740': s1_up_05,
    '3750': s1_down_01,
    '3760': s1_down_02,
    '3770': s1_down_03,
    '3780': s1_down_04,
    '3790': s1_down_05,
    
    '3800': intertrial_interval,

    # trial 6
    '4200': s1_up_01,
    '4210': s1_up_02,
    '4220': s1_up_03,
    '4230': s1_up_04,
    '4240': s1_up_05,
    '4250': s1_down_01,
    '4260': s1_down_02,
    '4270': s1_down_03,
    '4280': s1_down_04,
    '4290': s1_down_05,
    
    '4300': delay_period,

    '4500': s2_down_01,
    '4510': s2_down_02,
    '4520': s2_down_03,
    '4530': s2_down_04,
    '4540': s2_down_05,
    '4550': s2_up_01,
    '4560': s2_up_02,
    '4570': s2_up_03,
    '4580': s2_up_04,
    '4590': s2_up_05,

    '4600': intertrial_interval,

    # trial 7
    '5000': s1_up_01,
    '5010': s1_up_02,
    '5020': s1_up_03,
    '5030': s1_up_04,
    '5040': s1_up_05,
    '5050': s1_down_01,
    '5060': s1_down_02,
    '5070': s1_down_03,
    '5080': s1_down_04,
    '5090': s1_down_05,

    '5100': delay_period,

    '5300': s1_up_01,
    '5310': s1_up_02,
    '5320': s1_up_03,
    '5330': s1_up_04,
    '5340': s1_up_05,
    '5350': s1_down_01,
    '5360': s1_down_02,
    '5370': s1_down_03,
    '5380': s1_down_04,
    '5390': s1_down_05,
    
    '5400': intertrial_interval,

    # trial 8
    '5800': s1_up_01,
    '5810': s1_up_02,
    '5820': s1_up_03,
    '5830': s1_up_04,
    '5840': s1_up_05,
    '5850': s1_down_01,
    '5860': s1_down_02,
    '5870': s1_down_03,
    '5880': s1_down_04,
    '5890': s1_down_05,
    
    '5900': delay_period,

    '6100': s2_down_01,
    '6110': s2_down_02,
    '6120': s2_down_03,
    '6130': s2_down_04,
    '6140': s2_down_05,
    '6150': s2_up_01,
    '6160': s2_up_02,
    '6170': s2_up_03,
    '6180': s2_up_04,
    '6190': s2_up_05,

    '6200': intertrial_interval,

    # trial 9
    '6600': s1_up_01,
    '6610': s1_up_02,
    '6620': s1_up_03,
    '6630': s1_up_04,
    '6640': s1_up_05,
    '6650': s1_down_01,
    '6660': s1_down_02,
    '6670': s1_down_03,
    '6680': s1_down_04,
    '6690': s1_down_05,

    '6700': delay_period,

    '6900': s1_up_01,
    '6910': s1_up_02,
    '6920': s1_up_03,
    '6930': s1_up_04,
    '6940': s1_up_05,
    '6950': s1_down_01,
    '6960': s1_down_02,
    '6970': s1_down_03,
    '6980': s1_down_04,
    '6990': s1_down_05,
    
    '7000': intertrial_interval,

    # trial 10
    '7400': s1_up_01,
    '7410': s1_up_02,
    '7420': s1_up_03,
    '7430': s1_up_04,
    '7440': s1_up_05,
    '7450': s1_down_01,
    '7460': s1_down_02,
    '7470': s1_down_03,
    '7480': s1_down_04,
    '7490': s1_down_05,
    
    '7500': delay_period,

    '7700': s2_down_01,
    '7710': s2_down_02,
    '7720': s2_down_03,
    '7730': s2_down_04,
    '7740': s2_down_05,
    '7750': s2_up_01,
    '7760': s2_up_02,
    '7770': s2_up_03,
    '7780': s2_up_04,
    '7790': s2_up_05,

    '7800': intertrial_interval,

    # trial 11
    '8200': s1_up_01,
    '8210': s1_up_02,
    '8220': s1_up_03,
    '8230': s1_up_04,
    '8240': s1_up_05,
    '8250': s1_down_01,
    '8260': s1_down_02,
    '8270': s1_down_03,
    '8280': s1_down_04,
    '8290': s1_down_05,

    '8300': delay_period,

    '8500': s1_up_01,
    '8510': s1_up_02,
    '8520': s1_up_03,
    '8530': s1_up_04,
    '8540': s1_up_05,
    '8550': s1_down_01,
    '8560': s1_down_02,
    '8570': s1_down_03,
    '8580': s1_down_04,
    '8590': s1_down_05,
    
    '8600': intertrial_interval,

    # trial 12
    '9000': s1_up_01,
    '9010': s1_up_02,
    '9020': s1_up_03,
    '9030': s1_up_04,
    '9040': s1_up_05,
    '9050': s1_down_01,
    '9060': s1_down_02,
    '9070': s1_down_03,
    '9080': s1_down_04,
    '9090': s1_down_05,
    
    '9100': delay_period,

    '9300': s2_down_01,
    '9310': s2_down_02,
    '9320': s2_down_03,
    '9330': s2_down_04,
    '9340': s2_down_05,
    '9350': s2_up_01,
    '9360': s2_up_02,
    '9370': s2_up_03,
    '9380': s2_up_04,
    '9390': s2_up_05,

    '9400': intertrial_interval,
    
    ################### END OF DMS BLOCK ####################################
    
}


##- EoF -##
