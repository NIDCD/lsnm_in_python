# ============================================================================
#
#                            PUBLIC DOMAIN NOTICE
#
#       National Institute on Deafness and Other Communication Disorders
#
# This software/database is a "United States Government Work" under the 
# terms of the United States Copyright Act. It was written as part of 
# the author's official duties as a United States Government employee and 
# thus cannot be copyrighted. This software/database is freely available 
# to the public for use. The NIDCD and the U.S. Government have not placed 
# any restriction on its use or reproduction. 
#
# Although all reasonable efforts have been taken to ensure the accuracy 
# and reliability of the software and data, the NIDCD and the U.S. Government 
# do not and cannot warrant the performance or results that may be obtained 
# by using this software or data. The NIDCD and the U.S. Government disclaim 
# all warranties, express or implied, including warranties of performance, 
# merchantability or fitness for any particular purpose.
#
# Please cite the author in any work or product based on this material.
# 
# ==========================================================================



# ***************************************************************************
#
#   Large-Scale Neural Modeling software (LSNM)
#
#   Section on Brain Imaging and Modeling
#   Voice, Speech and Language Branch
#   National Institute on Deafness and Other Communication Disorders
#   National Institutes of Health
#
#   This file (plot_LSNM_connectivity.py) was created on April 18, 2016.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on April 18, 2016  
# **************************************************************************/

# plot_LSNM_connectivity.py
#
# Plot a heatmap showing the connectivity matrix of the LSNM modules contained in
# a neural network file of type JSON

from matplotlib import colors

import numpy as np
import matplotlib.pyplot as plt
import json

# where is the neural network stored?
neural_net_file = 'neuralnet.json'

# open the file that contains the neural net and dump its contents in a
# numpy array
with open(neural_net_file, 'r') as f:
    modules = json.load(f)

# how many modules do you want in the connectivity matrix?
net_size = 12

# initialize connectivity matrix
connectivity_matrix = np.zeros((net_size, net_size))

# create a map of module names to numbers to index the connectivity matrix
#dict = {'':''}
#list_of_modules = []
#index = 0
#for m in modules.keys():
#    dict.update({m: index})         # create a numeric index for each dictionary key
#    list_of_modules.append(m)       # create a list of modules to use as labels in heatmap
#    index = index + 1
dict = {'mgns': 0,
        'ea1u': 1,
        'ia1u': 1,
        'ea1d': 2,
        'ia1d': 2,
        'ea2u': 3,
        'ia2u': 3,
        'ea2c': 4,
        'ia2c': 4,
        'ea2d': 5,
        'ia2d': 5,
        'estg': 6,
        'istg': 6,
        'exfs': 7,
        'infs': 7,
        'efd1': 8,
        'ifd1': 8,
        'efd2': 9,
        'ifd2': 9,
        'exfr': 10,
        'infr': 10,
        'attv': 11,
        'atts': 11
        }

# declare list of modules in the same order of appearance as in the dictionary above:
list_of_modules = ['MGN', 'A1u', 'A1d', 'A2u', 'A2c', 'A2d', 'STG', 'FS', 'D1', 'D2', 'FR', 'Att']

# Traverse the modules data structure to find all of the connection weights and their
# destinations, and sum them up to fill out the connectivity matrix
for m in modules.keys():
    for x in range(modules[m][0]):
        for y in range(modules[m][1]):
            
            # we are going to do the following only for those units in the network that
            # have weights that project to other units elsewhere
            
            for w in modules[m][8][x][y][4]:
                
                # First, find outgoing weights for all destination units and (except
                # for those that do not
                # have outgoing weights, in which case do nothing) compute weight * value
                # at destination units
                dest_module = w[0]
                x_dest = w[1]
                y_dest = w[2]
                weight = w[3]

                # Assign the weight just found to its corresponding place in the connectivity
                # matrix, except in connection weights to itself (excluded from the heatmap
                # shown, as we only want cortico-cortical connections. For excitatory connections,
                # assign 1; for inhibitory connections, assign -1.
                if dict[m] == dict[dest_module]:
                    connectivity_matrix[dict[m]][dict[dest_module]] = 0.0
                else:
                    if dest_module[0] == 'e':
                        connectivity_matrix[dict[m]][dict[dest_module]] =  1.0
                    elif dest_module[0] == 'i':
                        connectivity_matrix[dict[m]][dict[dest_module]] = -1.0

fig, ax = plt.subplots()
cmap = colors.ListedColormap(['blue', 'white', 'red'])
heatmap = ax.pcolor(connectivity_matrix,cmap=cmap, alpha=0.7)
#cax = ax.pcolor(connectivity_matrix, cmap=cmap, vmin=-1, vmax=1)
#fig.colorbar(heatmap)

# Format
fig = plt.gcf()
#fig.set_size_inches(8, 11)

# turn off the frame
ax.set_frame_on(False)

# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(connectivity_matrix.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(connectivity_matrix.shape[1]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

#plt.xticks(range(0, len(list_of_modules), 1) )
#plt.yticks(range(0, len(list_of_modules), 1) )

ax.set_xticklabels(list_of_modules, minor=False)
ax.set_yticklabels(list_of_modules, minor=False)

# rotate the
plt.xticks(rotation=90)

ax.grid(False)

# Turn off all the ticks
ax = plt.gca()

for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False


print dict

print connectivity_matrix.size

print len(list_of_modules)

plt.show()
