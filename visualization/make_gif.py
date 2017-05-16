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
# contributed by Paul Corbitt on February 1 2017
############################################################################

import imageio 
# location of imageio github repository, note you will need to install imageio
# http://imageio.github.io/
# First you need to put in your list of images with the path to the image and file name
# make sure the files are in the order that you want them to be displayed
frame_names = ['brain_FC_matrix_screenshot_0.png',
               'brain_FC_matrix_screenshot_1.png',
               'brain_FC_matrix_screenshot_2.png',
               'brain_FC_matrix_screenshot_3.png',
               'brain_FC_matrix_screenshot_4.png',
               'brain_FC_matrix_screenshot_5.png',
               'brain_FC_matrix_screenshot_6.png',
               'brain_FC_matrix_screenshot_7.png',
               'brain_FC_matrix_screenshot_8.png',
               'brain_FC_matrix_screenshot_9.png']
number_of_frames = len(frame_names)
# the following loop writes out the gif to the present directory
# the gif display speed to 4 frames per second (fps)
with imageio.get_writer('./brain_FC_matrix_animated.gif', mode='I', fps = 4) as writer:
    for x in range(1,number_of_frames):
        image = imageio.imread(frame_names[x])
        writer.append_data(image)

