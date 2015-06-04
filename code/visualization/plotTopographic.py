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
#   This file (plotTopographic.py) was created on March 26, 2015.
#
#
#   Author: Antonio Ulloa. Last updated by Antonio Ulloa March 26, 2015  
# **************************************************************************/

# plotTopographic.py
#
# Plays a movie using output data files of visual delay-match-to-sample simulation

import numpy as np
import matplotlib.pyplot as plt

# Load data files
mgns = np.loadtxt('../../output/mgns.out')
efd1 = np.loadtxt('../../output/efd1.out')
efd2 = np.loadtxt('../../output/efd2.out')
ea1u = np.loadtxt('../../output/ea1u.out')
ea1d = np.loadtxt('../../output/ea1d.out')
ea2u = np.loadtxt('../../output/ea2u.out')
ea2d = np.loadtxt('../../output/ea2d.out')
ea2c = np.loadtxt('../../output/ea2c.out')
exfr = np.loadtxt('../../output/exfr.out')
exfs = np.loadtxt('../../output/exfs.out')
estg = np.loadtxt('../../output/estg.out')

fig = plt.figure(1)

plt.suptitle('SIMULATED NEURAL ACTIVITY')

# adds index of each array item to the value contained in the item
for (i,j), value in np.ndenumerate(mgns):
    mgns[i][j] = mgns[i][j] + j
    efd1[i][j] = efd1[i][j] + j
    efd2[i][j] = efd2[i][j] + j
    ea1u[i][j] = ea1u[i][j] + j
    ea1d[i][j] = ea1d[i][j] + j
    ea2u[i][j] = ea2u[i][j] + j
    ea2d[i][j] = ea2d[i][j] + j
    ea2c[i][j] = ea2c[i][j] + j
    exfr[i][j] = exfr[i][j] + j
    exfs[i][j] = exfs[i][j] + j
    estg[i][j] = estg[i][j] + j

# Render LGN array in a colormap
ax = plt.subplot(3,4,1)
plt.plot(mgns)
plt.title('MGN')
plt.ylim([38,69])

# Render EV1h array in a colormap
ax = plt.subplot(3,4,5)
plt.plot(ea1u)
plt.title('A1u')
plt.ylim([38,69])

# Render EV1v array in a colormap
ax = plt.subplot(3,4,9)
plt.plot(ea1d)
plt.title('A1d')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,2)
plt.plot(ea2u)
plt.title('A2u')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,6)
plt.plot(ea2d)
plt.title('A2d')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,10)
plt.plot(ea2c)
plt.title('A2c')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,3)
plt.plot(estg)
plt.title('STG')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,7)
plt.plot(exfs)
plt.title('FS')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,11)
plt.plot(efd1)
plt.title('FD1')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,4)
plt.plot(efd2)
plt.title('FD2')
plt.ylim([38,69])

# Render array in a colormap
ax = plt.subplot(3,4,8)
plt.plot(exfr)
plt.title('FR')
plt.ylim([38,69])

# Show the plot on the screen
plt.show()
