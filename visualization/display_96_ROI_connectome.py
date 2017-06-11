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
#   This file (display_Hagmanns_brain_connectivity.py) was created on July 18, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 18, 2017
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# display_Hagmanns_brain_connectivity.py
#
# Displays Hagmann's brain's 998-nodes plus LSNM nodes, along with the connections
# between Hagmann's nodes and LSNM nodes.
#
# The LSNM nodes to display can be defined in the code itself using Tailarach coordinates
# and can be motor, visual, or auditory. Perhaps in the future I will make this an option
# you can select from at run-time?
#

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

# build an array of TVB nodes that you want to look at closely to visualize what is
# connected to what
# Below are the node numbers for the TVB nodes where visual LSNM modules are embedded
nodes_to_be_examined = [79]

# Define the hypothetical Talairach locations of each LSNM visual modules
m1_lsnm = [-43, -7, 56]

# now, define the M1 coordinates in current connectome
m1 = []

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_96.zip")
centres = white_matter.centres

print white_matter

# Load one of the cortex 3d surface from TVB data files
CORTEX = surfaces.Cortex.from_file("cortex_80k/surface_80k.zip")

plot_surface(CORTEX, op=0.08)

# Threshold that will tell the visualization script whether to plot a given connection
# weight or ignore it
weight_threshold = 0.0

# Now plot the hypothetical locations of LSNM visual modules

# M1 node is green
#m1_module = mlab.points3d(m1[0],m1[1],m1[2], color=(0,0,1), scale_factor=5.)

# Plot the 96 nodes  (uncomment if needed for visualization purposes
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               color=(0.4, 0.4, 0.4),
                               scale_factor = 5.)


# ... now Plot the connections among the nodes
for tvb_node in nodes_to_be_examined:

    # draw node of interest in a diffent size and color
    m1_module = mlab.points3d(centres[tvb_node, 0],
                              centres[tvb_node, 1],
                              centres[tvb_node, 2],
                              color=(0,0,1), scale_factor=5.)
    
    print 'Node ', tvb_node, ' at ', centres[tvb_node], ' is connected to nodes: [', 
    
    # extract TVB node numbers that are connected to TVB node above by a value larger than
    # a given threshold
    tvb_conn = (white_matter.weights[tvb_node] > weight_threshold).nonzero()
    # get the connection that has the strongest weight
    #tvb_conn = np.argmax(white_matter.weights[tvb_node])
    # extract the numpy array from it
    tvb_conn = tvb_conn[0]
    #tvb_conn = [tvb_conn]
    
    for connected_node in tvb_conn:

        current_weight = white_matter.weights[tvb_node, connected_node]
        
        print connected_node, '(', current_weight, '),', 

        cxn = numpy.array([centres[connected_node],
                           centres[tvb_node]])

        connected = centres[connected_node]

        connections = mlab.plot3d(cxn[:, 0], cxn[:, 1], cxn[:, 2],
                                  color = (1, 0, 0),
                                  tube_radius=current_weight/3.0)
        
        connected = mlab.points3d(connected[0], connected[1], connected[2],
                                color=(0.75, 0.75, 0.75),
                                scale_factor = 5.)

    print ']'

# Finally, show everything on screen
mlab.show(stop=True)
