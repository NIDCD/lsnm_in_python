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
#   This file (gen_surround_inh.py) was created on December September 13, 2017.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on September 13 2017  
# **************************************************************************/

# gen_surround_inh.py
#
# Generates a weight file from excitatory mass of brain region A to inhibitory mass of
# brain region A that uses a "center excitation / surround inhibition" scheme
#
# The output file can be used in an LSNM simulation

origin_region      = "ev1v"
destination_region = "iv1v" 

# compose name of output file where weights will be saved
weight_file = origin_region + destination_region + ".w"      

center_value = 0.15              # value for center excitation
surround_value = 0.001            # value for surround inhibition

x_dim = 9                        # dimensions of output module in X axis
y_dim = 9                        # dimensions of output module in Y axis

file = open(weight_file, "w")    # open file for writing

file.write("% Weight file generated by gen_surround_inh.py \n")

file.write("\n")

file.write("Connect(" + origin_region + ", " + destination_region + ")  {\n")

for x in range(1, x_dim + 1):
    for y in range(1, y_dim + 1):
        file.write("  From:  (" + str(x) + ", " + str(y) + ")  {" + "\n")


        for neighbor_x in filter(lambda x: x>0 and x<=x_dim, range(x-1, x+2)):
            for neighbor_y in filter(lambda y: y>0 and y<=y_dim, range(y-1, y+2)):

                if (neighbor_x == x) and (neighbor_y == y):

                    # the following writes out the excitatory value at the center
                    file.write("    ([ " + str(x) + ", " + str(y) + "]  " +
                               str(center_value) + ")" + "\n")
                    
                else:
                    
                    # the following writes out the inhibitory values at the surrounding cells
                    file.write("    ([ " + str(neighbor_x) + ", " + str(neighbor_y) + "]  " +
                               str(surround_value) + ")" + "\n") 
                
        
        file.write("  }" + "\n")

file.write("}")

file.close()









