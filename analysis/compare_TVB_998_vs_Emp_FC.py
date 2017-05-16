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
#   This file (compare_TVB_998_vs_Emp_FC.py) was created on March 10 2017.
#
#
#   Author: Antonio Ulloa
#
#   Last updated by Antonio Ulloa on April 24 2017
#
# **************************************************************************/
#
# compare_TVB_998_vs_Emp_FC.py
#
# Reads several simulated Resting State BOLD fMRI simulations, computes functional
# connectivity matrices for each simulation, and compares them with an
# empirical functional connectivity (from Sporns and Honey) using a simple
# correlation value (between simulated and empirical FCs).

from tvb.simulator.lab import *

import numpy as np
import matplotlib.pyplot as plt

import scipy.io

from matplotlib import cm as CM

from scipy.stats import itemfreq

from sklearn.metrics import mean_squared_error

from mpl_toolkits.mplot3d import Axes3D


# define name of input file where Hagmann empirical data is stored (matlab file given to us
# by Olaf Sporns and Chris Honey
hagmann_data = 'DSI_release2_2011.mat'

# define name of output files where compressed, lo-res FC matrices will be stored.
tvb_fc_file = 'fc_mat_66x66.npy'

# define output file to store flat FC matrices
flat_tvb_fc_file = 'flat_fc_66x66.npy'

# define output file to store all correlation coefficients
corrcoeff_file = 'corr_coeffs.npy'

# load hagmann's structural connectivity matrix
hagmann_sc = connectivity.Connectivity.from_file("connectivity_998.zip")

# define the names of the input files where the correlation coefficients were stored
TVB_RS_FC  = ['output.RestingState.198_1_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0042/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0142/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0242/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0342/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0442/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0542/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0642/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0742/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0842/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_1_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_2_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_3_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_4_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_5_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_6_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_7_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_8_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_9_0.0942/xcorr_matrix_998_regions.npy',
              'output.RestingState.198_10_0.0942/xcorr_matrix_998_regions.npy'
             ]

# open matlab file that contains hagmann empirical data
hagmann_empirical = scipy.io.loadmat(hagmann_data)

ROIs = 998
lores_ROIs = 66

# open files that contain functional connectivities of simulated BOLD
i = 0
tvb_rs_fc = np.zeros([len(TVB_RS_FC), ROIs, ROIs])
for file in TVB_RS_FC: 
    tvb_rs_fc[i]  = np.load(TVB_RS_FC[i])
    i = i + 1

# we need to apply a Fisher Z transformation to the correlation coefficients,
# prior to averaging.
empirical_fc_mat_Z = np.arctanh(hagmann_empirical['COR_fMRI_average'])
tvb_rs_fc_Z = np.arctanh(tvb_rs_fc)

# initialize 66x66 matrices
empirical_fc_lowres_Z = np.zeros([lores_ROIs,lores_ROIs])
tvb_rs_fc_lowres_Z = np.zeros([len(TVB_RS_FC), lores_ROIs, lores_ROIs])


# subtract 1 from empirical labels array bc numpy arrays start with zero
hagmann_empirical['roi_lbls'][0] = hagmann_empirical['roi_lbls'][0] - 1

# compress 998x998 empirical FC matrix to 66x66 FC matrix by averaging
for i in range(0, ROIs):
    for j in range(0, ROIs):

        # extract low-res coordinates from hi-res empirical labels matrix
        x = hagmann_empirical['roi_lbls'][0][i]
        y = hagmann_empirical['roi_lbls'][0][j]

        empirical_fc_lowres_Z[x, y] += empirical_fc_mat_Z[i, j]

# count the number of times each lowres label appears in the hires matrix
freq_array = itemfreq(hagmann_empirical['roi_lbls'][0])

# divide each sum by the number of hi_res ROIs within each lowres ROI
for i in range(0, lores_ROIs):
    for j in range(0, lores_ROIs):
        total_freq = freq_array[i][1] * freq_array[j][1]
        empirical_fc_lowres_Z[i,j] = empirical_fc_lowres_Z[i,j] / total_freq 

# compress 998x998 simulated FC matrices to 66x66 FC matrices by averaging
for k in range(0, len(TVB_RS_FC)):
    for i in range(0, ROIs):
        for j in range(0, ROIs):

            # extract low-res coordinates from hi-res empirical labels matrix
            x = hagmann_empirical['roi_lbls'][0][i]
            y = hagmann_empirical['roi_lbls'][0][j]
            tvb_rs_fc_lowres_Z[k, x, y] += tvb_rs_fc_Z[k, i, j]

    # average the sum of each bucket dividing by no. of items in each bucket
    for i in range(0, lores_ROIs):
        for j in range(0, lores_ROIs):
            total_freq = freq_array[i][1] * freq_array[j][1]
            tvb_rs_fc_lowres_Z[k, i,j] = tvb_rs_fc_lowres_Z[k, i,j] / total_freq


# now, convert back to from Z to R correlation coefficients
empirical_fc_lowres = np.tanh(empirical_fc_lowres_Z)
tvb_rs_fc_lowres = np.tanh(tvb_rs_fc_lowres_Z)

# save the simulated FCs to a file 
np.save(tvb_fc_file, tvb_rs_fc_lowres)

# simulated RS FC needs to be rearranged prior to scatter plotting
#new_TVB_RS_FC = np.zeros([tvb_rs_fc_lowres.shape[0], lores_ROIs, lores_ROIs])
#for fc in range(0, tvb_rs_fc_lowres.shape[0]):
#    for i in range(0, lores_ROIs):
#        for j in range(0, lores_ROIs):
#
#            # extract labels of current ROI label from simulated labels list of simulated FC 
#            label_i = labels[i]
#            label_j = labels[j]
#            # extract index of corresponding ROI label from empirical FC labels list
#            emp_i = np.where(hagmann_empirical['anat_lbls'] == label_i)[0][0]
#            emp_j = np.where(hagmann_empirical['anat_lbls'] == label_j)[0][0]
#            new_TVB_RS_FC[fc, emp_i, emp_j] = tvb_rs_fc_lowres[fc, i, j]

# calculate binary SC matrix for a lo-res 66 ROI set (derived from the 998X998 sc)
bin_66_ROI_sc = np.zeros([lores_ROIs, lores_ROIs])
for i in range(0, ROIs):
    for j in range(0, ROIs):

        # extract low-res coordinates from hi-res empirical labels matrix
        x = hagmann_empirical['roi_lbls'][0][i]
        y = hagmann_empirical['roi_lbls'][0][j]

        if hagmann_sc.weights[i, j] > 0.:
            bin_66_ROI_sc[x, y] = 1

# apply mask to get rid of upper triangle, including main diagonal
mask = np.tri(tvb_rs_fc_lowres.shape[1], k=0)
mask = np.transpose(mask)

# apply mask to empirical FC matrix
empirical_fc_lowres_m = np.ma.array(empirical_fc_lowres, mask=mask)    # mask out upper triangle

# apply mask to binarized SC matrix
bin_66_ROI_sc_m = np.ma.array(bin_66_ROI_sc, mask=mask)     # mask out upper triangle

# apply mask to all simulated FC matrices
sim_rs_fc = np.ma.zeros([tvb_rs_fc.shape[0], lores_ROIs, lores_ROIs])
for fc in range(0, tvb_rs_fc.shape[0]):
    sim_rs_fc[fc] = np.ma.array(tvb_rs_fc_lowres[fc], mask=mask)    # mask out upper triangle

# flatten the empirical FC matriz
corr_mat_emp_FC = np.ma.ravel(empirical_fc_lowres_m)
corr_mat_emp_FC = np.ma.compressed(corr_mat_emp_FC)

# flatten the binarized SC matrix
flat_bin_66_ROI_sc = np.ma.ravel(bin_66_ROI_sc_m)
flat_bin_66_ROI_sc = np.ma.compressed(flat_bin_66_ROI_sc)

print 'Flat binarized 66 ROI SC: ', flat_bin_66_ROI_sc.shape

# flatten all simulated FC matrices
print 'Before ravel: ', sim_rs_fc[0].shape
print 'After ravel: ', np.ma.ravel(sim_rs_fc[0]).shape
print 'After compress: ', np.ma.compressed(np.ma.ravel(sim_rs_fc[0])).shape
flat_sim_rs_fc = np.zeros([tvb_rs_fc.shape[0], corr_mat_emp_FC.size])
for fc in range(0, tvb_rs_fc.shape[0]):
    # remove masked elements from cross-correlation matrix
    flat_sim_rs_fc[fc] = np.ma.compressed(np.ma.ravel(sim_rs_fc[fc]))

# save flat simulated FC matrices to file
np.save(flat_tvb_fc_file, flat_sim_rs_fc)

# now, construct a mask using the binarized 66 ROI SC array.
# THis mask will be used to calculate correlations between FC connectivity
# matrices only when a structural connection exists between two regions
sc_mask = flat_bin_66_ROI_sc == 0                              # create mask
corr_mat_emp_FC_m = np.ma.array(corr_mat_emp_FC, mask=sc_mask) # apply mask
flat_emp_fc = np.ma.compressed(corr_mat_emp_FC_m)              # remove masked elements

print 'Mask for flat binarized 66 ROI SC matrix: ', sc_mask

    
# calculate correlation coefficients and MSE (mean squared error) for
# empirical FC matrix vs each simulated FC matrix
mse = np.zeros(tvb_rs_fc.shape[0])
r   = np.zeros(tvb_rs_fc.shape[0])
for fc in range(0, tvb_rs_fc.shape[0]):
    flat_sim_fc_m = np.ma.array(flat_sim_rs_fc[fc], mask=sc_mask)
    flat_sim_fc = np.ma.compressed(flat_sim_fc_m)
    r[fc]   = np.corrcoef(flat_emp_fc, flat_sim_fc)[1,0]
    mse[fc] = mean_squared_error(flat_emp_fc, flat_sim_fc)

# save correlation coefficients to a file
np.save(corrcoeff_file, r)

# print all correlations:
print 'Correlations empirical vs simulated: '
print r

# print all MSE's:
print 'Mean Squared Errors between empirical and simulated: '
print mse        

    
# calculates the index of the maximum correlation value
max_r = np.argmax(r)
print ' Best correlation found was: ', r[max_r], ', element number ', max_r

# initialize figure to plot heatmap of correlations btw empirical and simulated fc
fig=plt.figure('Correlation btw empirical and simulated fc as a function of two parameters')
ax = fig.add_subplot(111)
cmap = CM.get_cmap('jet', 10)
r_r = r.reshape(10, 10)
r_ud = np.flipud(r_r)
cax = ax.imshow(r_ud, interpolation='nearest', cmap=cmap)
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_yticks([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
ax.set_yticklabels(['.0042', '.0142', '.0242', '.0342', '.0442',
                    '.0542', '.0642', '.0742', '.0842', '.0942'])
plt.xlabel('Conduction speed')
plt.ylabel('Global coupling strength')
color_bar=plt.colorbar(cax)
fig.savefig('corrs_empirical_vs_simulated_rs_fc.png')

# initialize figure to plot FC
fig=plt.figure('Functional Connectivity Matrix of empirical BOLD (66 ROIs)')
ax = fig.add_subplot(111)
cmap = CM.get_cmap('jet', 10)
empirical_fc_lowres = np.asarray(empirical_fc_lowres)
cax = ax.imshow(empirical_fc_lowres, vmin=-0.4, vmax=1.0, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

# plot the simulated FC with the highest correlation coefficient
fig=plt.figure('Functional Connectivity Matrix of simulated BOLD (66 ROIs)')
ax = fig.add_subplot(111)
cmap = CM.get_cmap('jet', 10)
cax = ax.imshow(tvb_rs_fc_lowres[max_r], vmin=-0.4, vmax=1.0, interpolation='nearest', cmap=cmap)
ax.grid(False)
color_bar=plt.colorbar(cax)

# plot the structural connectivity of the empirical Hagmann's connectome
fig = plt.figure('Hagmanns Structural Connectivity Matrix (998 ROIs)')
ax = fig.add_subplot(111)
cax2 = ax.imshow(hagmann_sc.weights, cmap=cmap)
color_bar2 = plt.colorbar(cax2)

# now, plot the binary 66 ROI sc matrix!
fig = plt.figure('Binarized sctructural connectivity matrix (66 ROIs)')
ax = fig.add_subplot(111)
cax3 = ax.imshow(bin_66_ROI_sc, cmap=cmap)
color_bar3 = plt.colorbar(cax3)

# scatter plot of hi-res (998 ROIs) empirical structural connectivity vs empirical functional connectivity
fig=plt.figure('Empirical SC vs empirical FC')
# apply mask to get rid of upper triangle, including main diagonal
hr_mask = np.tri(hagmann_sc.weights.shape[0], k=0)                   # create mask
hr_mask = np.transpose(hr_mask)                                # turn mask upside down
hires_sc_m = np.ma.array(hagmann_sc.weights, mask=hr_mask)     # apply mask
hires_fc_m = np.ma.array(hagmann_empirical['COR_fMRI_average'], mask=hr_mask)
flat_sc = np.ma.ravel(hires_sc_m)                              # flatten array
flat_fc = np.ma.ravel(hires_fc_m)
flat_sc = np.ma.compressed(flat_sc)                            # remove masked items
flat_fc = np.ma.compressed(flat_fc)

# mask to zero-value (absent structural connections) in the SC matrix
h_mask = flat_sc == 0.
masked_sc = np.ma.array(flat_sc, mask=h_mask)
flat_sc = np.ma.compressed(masked_sc)                      # remove masked elements
masked_fc = np.ma.array(flat_fc, mask=h_mask)
flat_fc = np.ma.compressed(masked_fc)                      # remove masked elements
plt.scatter(flat_sc, flat_fc)
plt.xlabel('Empirical Structural Connectivity')
plt.ylabel('Empirical Functional Connectivity')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_sc, flat_fc, 1)
plt.plot(flat_sc, m*flat_sc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
cc = np.corrcoef(flat_sc, flat_fc)[1,0]
plt.text(0.9, -0.3, 'r=' + '{:.2f}'.format(cc))

#################################################################
# scatter plot of lo-res (66 ROI) empirical FC vs best matched simulated FC 
# initialize figure to plot correlations between empirical and simulated FC
fig=plt.figure('Empirical vs Simulated FC (66 ROIs)')
flat_sim_fc_m = np.ma.array(flat_sim_rs_fc[max_r], mask=sc_mask)   # mask where struct. connections absent
flat_sim_fc = np.ma.compressed(flat_sim_fc_m)                      # remove masked elements
plt.scatter(flat_emp_fc, flat_sim_fc)
plt.xlabel('Empirical FC')
plt.ylabel('Model FC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_emp_fc, flat_sim_fc, 1)
plt.plot(flat_emp_fc, m*flat_emp_fc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
cc = np.corrcoef(flat_emp_fc, flat_sim_fc)[1,0]
plt.text(0.5, -0.1, 'r=' + '{:.2f}'.format(cc))
fig.savefig('best_corr_emp_vs_sim_rs_fc.png')

#################################################################
# scatter plot of lo-res (66 ROI) empirical SC vs best matched simulated FC 
# initialize figure to plot correlations between empirical and simulated FC
fig=plt.figure('Structural SC vs Simulated FC (66 ROIs)')
flat_sc_m = np.ma.array(hagmann_sc_lowres, mask=sc_mask)   # mask where struct. connections absent
flat_sc = np.ma.compressed(flat_sc_m)                      # remove masked elements
plt.scatter(flat_sc, flat_sim_fc)
plt.xlabel('Empirical FC')
plt.ylabel('Empirical SC')

# fit scatter plot with np.polyfit
m, b = np.polyfit(flat_sc, flat_sim_fc, 1)
plt.plot(flat_sc, m*flat_sc + b, '-', color='red')

# calculate correlation coefficient and display it on plot
cc = np.corrcoef(flat_sc, flat_sim_fc)[1,0]
plt.text(0.5, -0.1, 'r=' + '{:.2f}'.format(cc))
fig.savefig('sc_vs_sim_rs_fc.png')


# finally, show all figures!
plt.show()
