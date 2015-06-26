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
#   This file (plot_TVB_nodes.py) was created on June 17 2015
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa on June 17 2015  
# **************************************************************************/

# plot_TVB_nodes.py
#
# Plot electrical activity in given TVB nodes in a preprocessed TVB simulation
# file with extension *.npy

import numpy as np
import matplotlib.pyplot as pl

RawData = np.load("../simulator/wilson_cowan_brain_998_nodes.npy")

pl.plot(RawData[:,0,345])
pl.plot(RawData[:,0,393])
pl.plot(RawData[:,0,413])
pl.plot(RawData[:,0,74])
pl.grid(True)

pl.show()
