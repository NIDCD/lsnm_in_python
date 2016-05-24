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
#   This file (plot_weights_among_given_nodes.py) was created on May 12, 2016.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on May 12, 2016
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# plot_weights_among_given_nodes.py
#
# Displays Hagmann's brain's 998-nodes plus the weight among given nodes
# It also prints out what the connection weights are among those given
# connectome nodes.
#

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

# build an array of TVB nodes that you want to look at closely to visualize what is
# connected to what
# Below are the node numbers for the TVB nodes where visual LSNM modules are embedded
# the following contains all nodes grouped by ROI
#nodes_list = [range(344, 354),    # rPCAL, where V1/V2 is embedded
#              range(390, 412),    # rFUS, where V4 is embedded
#              range(412, 418),    # rPARH, where IT is embedded
#              range(47, 57),      # rPOPE, where FS is embedded
#              range(57, 79),      # rRMF, where D1 is embedded
#              range(39, 47),      # rPTRI, where D2 is embedded
#              range(125, 138)]    # rCMF, whenre FR is embedded

nodes_list = [range(344, 350),    #  V1/V2 ROI
              range(390, 396),    #  V4 ROI
              range(412, 418),    #  IT ROI
              range(47, 53),      #  FS ROI
              range(73, 79),      #  D1 ROI
              range(39, 45),      #  D2 ROI
              range(125, 131)]    #  FR ROI

#nodes_list = [range(345, 346),    #  V1/V2 ROI
#              range(393, 394),    #  V4 ROI
#              range(413, 414),    #  IT ROI
#              range(47, 48),      #  FS ROI
#              range(74, 75),      #  D1 ROI
#              range(41, 42),      #  D2 ROI
#              range(125, 126)]    #  FR ROI

nodes_grouped_by_region = np.array(nodes_list)

# the following contains all nodes with no grouping
nodes_to_be_examined = np.hstack(nodes_grouped_by_region)

print 'Nodes grouped by region:', nodes_grouped_by_region
print 'All nodes with no grouping:', nodes_to_be_examined

nodes  = np.array([345,          # V1/V2
                   393,          # V4
                   413,          # IT
                   47,           # FS
                   74,           # D1
                   41,           # D2
                   125])         # FR

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

# V1 host node is yellow
v1_module = mlab.points3d(centres[nodes_grouped_by_region[0], 0],
                          centres[nodes_grouped_by_region[0], 1],
                          centres[nodes_grouped_by_region[0], 2],
                          color=(1, 1, 0),scale_factor = 4.)

# V4 node is green
v4_module = mlab.points3d(centres[nodes_grouped_by_region[1], 0],
                          centres[nodes_grouped_by_region[1], 1],
                          centres[nodes_grouped_by_region[1], 2],
                          color=(0, 1, 0),scale_factor = 4.)

# IT node is blue
it_module = mlab.points3d(centres[nodes_grouped_by_region[2], 0],
                          centres[nodes_grouped_by_region[2], 1],
                          centres[nodes_grouped_by_region[2], 2],
                          color=(0, 0, 1),scale_factor = 4.)

# FS node is orange
fs_module = mlab.points3d(centres[nodes_grouped_by_region[3], 0],
                          centres[nodes_grouped_by_region[3], 1],
                          centres[nodes_grouped_by_region[3], 2],
                          color=(1, 0.5, 0),scale_factor = 4.)

# D1 node is red
d1_module = mlab.points3d(centres[nodes_grouped_by_region[4], 0],
                          centres[nodes_grouped_by_region[4], 1],
                          centres[nodes_grouped_by_region[4], 2],
                          color=(1, 0, 0),scale_factor = 4.)

# D2 node is magenta (or is it pink?)
d2_module = mlab.points3d(centres[nodes_grouped_by_region[5], 0],
                          centres[nodes_grouped_by_region[5], 1],
                          centres[nodes_grouped_by_region[5], 2],
                          color=(1, 0, 1),scale_factor = 4.)

# FR node is purple
fr_module = mlab.points3d(centres[nodes_grouped_by_region[6], 0],
                          centres[nodes_grouped_by_region[6], 1],
                          centres[nodes_grouped_by_region[6], 2],
                          color=(0.5, 0, 0.5),scale_factor = 4.)


# ... now Plot the connections weights among the nodes
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
        tvb_node_region = next(((i, node.index(tvb_node)) for i, node in enumerate(nodes_list) if tvb_node in node), None)[0]
        tvb_conn_region = next(((i, node.index(connected_node)) for i, node in enumerate(nodes_list) if connected_node in node), None)[0]
        
        #tvb_node_region = np.where(nodes_list==tvb_node)[0][0]
        #tvb_conn_region = np.where(nodes_list==connected_node)[0][0]
        
        # Display the current connection ONLY IF connected nodes do not belong to
        # the same region:
        if tvb_node_region != tvb_conn_region:

            print 'Node ', tvb_node, ' is connected to node ', connected_node
            
            cxn = numpy.array([centres[connected_node],
                               centres[tvb_node]])

            connections = mlab.plot3d(cxn[:, 0], cxn[:, 1], cxn[:, 2],
                                      color = (0, 0, 0),
                                      tube_radius=0.3)
        
# Finally, show everything on screen
mlab.show(stop=True)
