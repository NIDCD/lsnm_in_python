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
#   This file (display_connectivity_among_given_nodes.py) was created on September 15, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on September 15, 2015
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# display_connectivity_among_given_nodes.py
#
# Displays Hagmann's brain's 998-nodes plus the connections among given nodes
# and it also prints out what the connection weights are among those given
# connectome nodes
#
# Note: this is just for visualization purposes, and it is only showing a
# "minimal network" among a set of nodes. 

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

# build an array of TVB nodes that you want to look at closely to visualize what is
# connected to what
# Below are the node numbers for the TVB nodes where visual LSNM modules are embedded
# the following contains all nodes grouped by ROI
nodes_grouped_by_ROI = np.array([[344, 344, 344, 344, 344, 344],
                                 [390, 390, 390, 390, 390, 390],
                                 [423, 425, 425, 425, 425, 425],
                                 [430, 430, 430, 430, 430, 430],
                                 [41,  41,  41,  41,  41,  41],
                                 [73,  74,  75,  73,  73,  73],
                                 [47,  47,  47,  47,  47,  52],
                                 [125, 125, 125, 125, 125, 125]])
#nodes_grouped_by_ROI = np.array([[344, 350, 351, 352, 353, 344],
#                                 [390, 429, 430, 431, 432, 433],
#                                 [423, 423, 423, 423, 423, 425],
#                                 [47,  48,  49,  50,  51,  52],
#                                 [73,  74,  75,  76,  77,  78],
#                                 [39,  40,  41,  42,  43,  44],
#                                 [125, 126, 127, 128, 129, 130]])


# the following contains all nodes with no grouping
nodes_to_be_examined = nodes_grouped_by_ROI.flatten()

# now, define the TVB nodes that are closest to the visual LSNM module locations above
v1 = [18, -91, 2]       # node 344
v4 = [23, -83, -4]      # node 390
it = [43, -60, 1]       # node 423
fs = [47, 19, 9]        # node 47
d1 = [43, 29, 21]       # node 74
d2 = [42, 39, 2]        # node 41 
fr = [29, 25, 40]       # node 125 

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_998.zip")
centres = white_matter.centres

# Plot the 998 nodes of Hagmann's brain
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               color=(0.5, 0.5, 0.5),
                               scale_factor = 1.)

# Now plot the hypothetical locations of LSNM visual modules

# V1 node is yellow
v1_module = mlab.points3d(v1[0],v1[1],v1[2],color=(1, 1, 0),scale_factor = 8.)

# V4 node is green
v4_module = mlab.points3d(v4[0],v4[1],v4[2],color=(0, 1, 0),scale_factor = 8.)

# IT node is blue
it_module = mlab.points3d(it[0],it[1],it[2],color=(0, 0, 1),scale_factor = 8.)

# FS node is orange
fs_module = mlab.points3d(fs[0],fs[1],fs[2],color=(1, 0.5, 0),scale_factor = 8.)

# D1 node is red
d1_module = mlab.points3d(d1[0],d1[1],d1[2],color=(1, 0, 0),scale_factor = 8.)

# D2 node is magenta (or is it pink?)
d2_module = mlab.points3d(d2[0],d2[1],d2[2],color=(1, 0, 1),scale_factor = 8.)

# FR node is purple
fr_module = mlab.points3d(fr[0],fr[1],fr[2],color=(0.5, 0, 0.5),scale_factor = 8.)


# ... now Plot the connections among the nodes, but only if those connections go
# to a different ROI
for tvb_node in nodes_to_be_examined:

    # extract TVB node numbers that are connected to TVB node above
    tvb_conn = np.nonzero(white_matter.weights[tvb_node])
    # extract the numpy array from it
    tvb_conn = tvb_conn[0]
    # now get rid of connected nodes that are not within the group of nodes that we are
    # interested in examining
    tvb_conn = np.intersect1d(tvb_conn, nodes_to_be_examined)
    
    
    for connected_node in tvb_conn:

        # first, check whether the two nodes that we are connecting belong to same
        # ROI:
        tvb_node_ROI = np.where(nodes_grouped_by_ROI==tvb_node)[0][0]
        tvb_conn_ROI = np.where(nodes_grouped_by_ROI==connected_node)[0][0]

        
        # Display the current connection ONLY IF connected nodes do not belong to
        # the same ROI:
        if tvb_node_ROI != tvb_conn_ROI:

            print 'Node ', tvb_node, ' is connected to node ', connected_node
            
            cxn = numpy.array([centres[connected_node],
                           centres[tvb_node]])

            connections = mlab.plot3d(cxn[:, 0], cxn[:, 1], cxn[:, 2],
                                      color = (0, 0, 0),
                                      tube_radius=0.3)
        
# Finally, show everything on screen
mlab.show(stop=True)
