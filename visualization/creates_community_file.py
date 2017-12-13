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
#   This file (creates_community_file.py) was created on December 6 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on December 6 2017
#
# **************************************************************************/

import numpy as np

import csv

import pandas as pd

hires_ROIs = 998

# the following ranges define the location of the nodes within a given ROI in Hagmann's brain.
# They were taken from the excel document:
#       "Location of visual LSNM modules within Connectome.xlsx"
# Extracted from The Virtual Brain Demo Data Sets

roi_dict = {
    'rLOF'  : range(  0,  19),    
    'rPORB' : range( 19,  25),          
    'rFP'   : range( 25,  27),          
    'rMOF'  : range( 27,  39),          
    'rPTRI' : range( 39,  47),          
    'rPOPE' : range( 47,  57),          
    'rRMF'  : range( 57,  79),          
    'rSF'   : range( 79, 125),          
    'rCMF'  : range(125, 138),          
    'rPREC' : range(138, 174),          
    'rPARC' : range(174, 186),          
    'rRAC'  : range(186, 190),          
    'rCAC'  : range(190, 194),          
    'rPC'   : range(194, 201),          
    'rISTC' : range(201, 209),          
    'rPSTC' : range(209, 240),          
    'rSMAR' : range(240, 256),          
    'rSP'   : range(256, 283),          
    'rIP'   : range(283, 311),          
    'rPCUN' : range(311, 334),          
    'rCUN'  : range(334, 344),          
    'rPCAL' : range(344, 354),          
    'rLOCC' : range(354, 373),          
    'rLING' : range(373, 390),          
    'rFUS'  : range(390, 412),          
    'rPARH' : range(412, 418),          
    'rENT'  : range(418, 420),          
    'rTP'   : range(420, 423),          
    'rIT'   : range(423, 442),          
    'rMT'   : range(442, 462),          
    'rBSTS' : range(462, 469),          
    'rST'   : range(469, 497),          
    'rTT'   : range(497, 500),
    'lLOF'  : range(500, 520),
    'lPORB' : range(520, 526),
    'lFP'   : range(526, 528),
    'lMOF'  : range(528, 540),
    'lPTRI' : range(540, 547),
    'lPOPE' : range(547, 558),
    'LRMF'  : range(558, 577),
    'lSF'   : range(577, 627),
    'lCMF'  : range(627, 640),
    'lPREC' : range(640, 676),
    'lPARC' : range(676, 687),
    'lRAC'  : range(687, 691),
    'lCAC'  : range(691, 695),
    'lPC'   : range(695, 702),
    'lISTC' : range(702, 710),
    'lPSTC' : range(710, 740),
    'lSMAR' : range(740, 759),
    'lSP'   : range(759, 786),
    'lIP'   : range(786, 811),
    'lPCUN' : range(811, 834),
    'lCUN'  : range(834, 842),
    'lPCAL' : range(842, 851),
    'lLOC'  : range(851, 873),
    'lLING' : range(873, 889),
    'lFUS'  : range(889, 911),
    'lPARH' : range(911, 917),
    'lENT'  : range(917, 920),
    'lTP'   : range(920, 924),
    'lIT'   : range(924, 941),
    'lMT'   : range(941, 960),
    'lBSTS' : range(960, 965),
    'lST'   : range(965, 994),
    'lTT'   : range(994, 998)
}

# declares membership of each one of the 66 areas above in one of six modules as defined by Hagmann et al (2008)
# See Table S2 from Supplementary material from Lee et al, 2016, Neuroimage
CI = [6,6,6,6,6,6,6,6,6,6,2,6,2,2,2,4,4,4,4,2,1,1,4,1,4,4,4,4,4,4,4,4,4,
      5,5,5,5,5,5,5,5,5,5,2,5,2,2,2,3,3,3,3,1,1,1,3,1,3,1,3,3,3,3,3,3,3]

ROI_array = ['rLOF','rPORB','rFP','rMOF','rPTRI','rPOPE','rRMF','rSF','rCMF','rPREC',
             'rPARC','rRAC','rCAC','rPC','rISTC','rPSTC', 'rSMAR','rSP','rIP','rPCUN',          
             'rCUN','rPCAL','rLOCC','rLING','rFUS','rPARH','rENT','rTP','rIT','rMT',
             'rBSTS', 'rST', 'rTT', 'lLOF','lPORB', 'lFP', 'lMOF','lPTRI','lPOPE','LRMF',
             'lSF','lCMF','lPREC','lPARC','lRAC','lCAC','lPC','lISTC','lPSTC','lSMAR',
             'lSP','lIP','lPCUN','lCUN','lPCAL','lLOC','lLING','lFUS','lPARH','lENT',
             'lTP','lIT','lMT','lBSTS','lST','lTT']

# declares membership of each one of the 998 areas above in one of six modules as defined by Hagmann et al (2008)
CI_hires = np.zeros(hires_ROIs, dtype=np.int)
for idx, lowres_node in enumerate(ROI_array):
    CI_hires[roi_dict[lowres_node]] = CI[idx]

indexes = map(str, range(1, hires_ROIs+1))     # convert list of 1-998 to strings
indexes = ['i'+ e for e in indexes]            # append a string to hack it into CSV
df = pd.DataFrame(CI_hires, index=indexes)
df.to_csv('community_membership.csv')
