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
#   This file (display_66_ROI_FC) was created on January 16, 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on January 16, 2017
#   Based on: display_sensor_locations.py by Paula Sanz-Leon (TVB team)
# **************************************************************************/

# display_66_ROI_FC.py
#
# Displays Hagmann's brain's 66-nodes (low-res ROIs)

from tvb.simulator.lab import *
from tvb.simulator.plot.tools import mlab

# Load connectivity from Hagmann's brain
white_matter = connectivity.Connectivity.from_file("connectivity_66.zip")
centres = white_matter.centres

# Plot the 66 ROIs of Hagmann's brain
region_centres = mlab.points3d(centres[:, 0], 
                               centres[:, 1], 
                               centres[:, 2],
                               color=(0.5, 0.5, 0.5),
                               scale_factor = 5.)

#        connections = mlab.plot3d(cxn[:, 0], cxn[:, 1], cxn[:, 2],
#                                  color = (0, 0, 0),
#                                  tube_radius=0.5)
        
#        connected = mlab.points3d(connected[0], connected[1], connected[2],
#                                color=(0.75, 0.75, 0.75),
#                                scale_factor = 8.)

# Finally, show everything on screen
mlab.show(stop=True)
