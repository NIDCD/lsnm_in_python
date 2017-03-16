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
#   This file (compare_TVB_vs_Emp_FC.py) was created on March 7 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on March 7 2017
#
# **************************************************************************/
#
# compare_TVB_vs_Emp_FC.py
#
# Reads several simulated Resting State BOLD fMRI simulations, computes functional
# connectivity matrices for each simulation, and compares them with an
# empirical functional connectivity (from Sporns and Honey) using a simple
# correlation value (between simulated and empirical FCs).

import numpy as np
import matplotlib.pyplot as plt

import scipy.io

from matplotlib import cm as CM

from scipy.stats import itemfreq



# define name of input file where Hagmann empirical data is stored (matlab file given to us
# by Olaf Sporns and Chris Honey
hagmann_data = 'DSI_release2_2011.mat'

# define the names of the input files where the correlation coefficients were stored
TVB_RS_FC  = ['output.RestingState.198_1_0.0242/xcorr_matrix_66_regions.npy',
              'output.RestingState.198_5_0.0242/xcorr_matrix_66_regions.npy'
              ]

# declare ROI labels as ordered in the simulated FC files
labels =  [' rLOF',     
           'rPORB',         
           '  rFP',          
           ' rMOF',          
           'rPTRI',          
           'rPOPE',          
           ' rRMF',          
           '  rSF',          
           ' rCMF',          
           'rPREC',          
           'rPARC',          
           ' rRAC',          
           ' rCAC',          
           '  rPC',          
           'rISTC',          
           'rPSTC',          
           'rSMAR',          
           '  rSP',          
           '  rIP',          
           'rPCUN',          
           ' rCUN',          
           'rPCAL',          
           'rLOCC',          
           'rLING',          
           ' rFUS',          
           'rPARH',          
           ' rENT',          
           '  rTP',          
           '  rIT',          
           '  rMT',          
           'rBSTS',          
           '  rST',          
           '  rTT',
           ' lLOF',
           'lPORB',
           '  lFP',
           ' lMOF',
           'lPTRI',
           'lPOPE',
           ' lRMF',
           '  lSF',
           ' lCMF',
           'lPREC',
           'lPARC',
           ' lRAC',
           ' lCAC',
           '  lPC',
           'lISTC',
           'lPSTC',
           'lSMAR',
           '  lSP',
           '  lIP',
           'lPCUN',
           ' lCUN',
           'lPCAL',
           'lLOCC',
           'lLING',
           ' lFUS',
           'lPARH',
           ' lENT',
           '  lTP',
           '  lIT',
           '  lMT',
           'lBSTS',
           '  lST',
           '  lTT'
]            


# open matlab file that contains hagmann empirical data
hagmann_empirical = scipy.io.loadmat(hagmann_data)

# open files that contain functional connectivities
i = 0
tvb_rs_fc = np.zeros([len(TVB_RS_FC), ROIs, ROIs])
for file in TVB_RS_FC: 
    tvb_rs_fc[i]  = np.load(TVB_RS_FC[i])
    i = i + 1

# extract total number of ROIs from one of the FC matrices
ROIs = tvb_rs_fc[0].shape[0]
    
# we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
empirical_fc_mat_Z = np.arctanh(hagmann_empirical['COR_fMRI_average'])

# initialize 66x66 matrix
empirical_fc_lowres_Z = np.zeros([ROIs,ROIs])

# subtract 1 from labels array bc numpy arrays start with zero
hagmann_empirical['roi_lbls'][0] = hagmann_empirical['roi_lbls'][0] - 1

# compress 998x998 empirical FC matrix to 66x66 FC matrix by averaging
for i in range(0,998):
    for j in range(0,998):

        # extract low-res coordinates from hi-res labels matrix
        x = hagmann_empirical['roi_lbls'][0][i]
        y = hagmann_empirical['roi_lbls'][0][j]

        empirical_fc_lowres_Z[x, y] += empirical_fc_mat_Z[i, j]

# count the number of times each lowres label appears in the hires matrix
freq_array = itemfreq(hagmann_empirical['roi_lbls'][0])

# divide each sum by the number of hires ROIs within each lowres ROI
for i in range(0,66):
    for j in range(0,66):
        total_freq = freq_array[i][1] * freq_array[j][1]
        empirical_fc_lowres_Z[i,j] = empirical_fc_lowres_Z[i,j] / total_freq 

# now, convert back to from Z to R correlation coefficients
empirical_fc_lowres = np.tanh(empirical_fc_lowres_Z)

# initialize figure to plot simulated FC
fig=plt.figure('Functional Connectivity Matrix of empirical BOLD (66 ROIs)')
ax = fig.add_subplot(111)
cmap = CM.get_cmap('jet', 10)
empirical_fc_lowres = np.asarray(empirical_fc_lowres)
cax = ax.imshow(empirical_fc_lowres, vmin=-0.4, vmax=1.0, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

# simulated RS FC needs to be rearranged prior to scatter plotting
new_TVB_RS_FC = np.zeros([tvb_rs_fc.shape[0], ROIs, ROIs])
for fc in range(0, tvb_rs_fc.shape[0]):
    for i in range(0, ROIs):
        for j in range(0, ROIs):

            # extract labels of current ROI label from simulated labels list of simulated FC 
            label_i = labels[i]
            label_j = labels[j]
            # extract index of corresponding ROI label from empirical FC labels list
            emp_i = np.where(hagmann_empirical['anat_lbls'] == label_i)[0][0]
            emp_j = np.where(hagmann_empirical['anat_lbls'] == label_j)[0][0]
            new_TVB_RS_FC[fc, emp_i, emp_j] = tvb_rs_fc[fc, i, j]

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(new_TVB_RS_FC.shape[1], k=0)
mask = np.transpose(mask)

# apply mask to empirical FC matrix
empirical_fc_lowres = np.ma.array(empirical_fc_lowres, mask=mask)    # mask out upper triangle

# apply mask to all simulated FC matrices
sim_rs_fc = np.ma.zeros([tvb_rs_fc.shape[0], ROIs, ROIs])
for fc in range(0, tvb_rs_fc.shape[0]):
    sim_rs_fc[fc] = np.ma.array(new_TVB_RS_FC[fc], mask=mask)    # mask out upper triangle

# flatten the empirical FC matriz
corr_mat_emp_FC = np.ma.ravel(empirical_fc_lowres)
corr_mat_emp_FC = np.ma.compressed(corr_mat_emp_FC)

# flatten all simulated FC matrices
print 'Before ravel: ', sim_rs_fc[0].shape
print 'After ravel: ', np.ma.ravel(sim_rs_fc[0]).shape
print 'After compress: ', np.ma.compressed(np.ma.ravel(sim_rs_fc[0])).shape
flat_sim_rs_fc = np.zeros([tvb_rs_fc.shape[0], corr_mat_emp_FC.size])
for fc in range(0, tvb_rs_fc.shape[0]):
    # remove masked elements from cross-correlation matrix
    flat_sim_rs_fc[fc] = np.ma.compressed(np.ma.ravel(sim_rs_fc[fc]))

# calculate correlation coefficients for empirical FC matrix vs each simulated FC matrix
r = np.zeros(tvb_rs_fc.shape[0])
for fc in range(0, tvb_rs_fc.shape[0]):
    r[fc] = np.corrcoef(corr_mat_emp_FC, flat_sim_rs_fc[fc])[1,0]

# print all correlations:
print 'Correlations empirical vs simulated: '
print r
    
# calculates the index of the maximum correlation value
max_r = np.argmax(r)
print ' Best correlation found was: ', r[max_r], ', element number ', max_r

# plot the simulated FC with the highest correlation coefficient
fig=plt.figure('Functional Connectivity Matrix of simulated BOLD (66 ROIs)')
ax = fig.add_subplot(111)
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(new_TVB_RS_FC[max_r], vmin=-0.4, vmax=1.0, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

# scatter plot of empirical FC vs best matched simulated FC 
# initialize figure to plot correlations between empirical and simulated FC
fig=plt.figure('Empirical vs Simulated FC')
plt.scatter(corr_mat_emp_FC, flat_sim_rs_fc[max_r])
plt.xlabel('Empirical FC')
plt.ylabel('Model FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(corr_mat_emp_FC, flat_sim_rs_fc[max_r], 1)
plt.plot(corr_mat_emp_FC, m*corr_mat_emp_FC + b, '-', color='red')

# calculate correlation coefficient and display it on plot
cc = np.corrcoef(corr_mat_emp_FC, flat_sim_rs_fc[max_r])[1,0]
plt.text(0.5, 0.3, 'r=' + '{:.2f}'.format(cc))


# finally, show all figures!
plt.show()
