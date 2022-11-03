# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:14:57 2022

@author: Innes
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from draw_matches import draw_matches_error
import sys

'''Computes the error for a set of matches given a ground truth (ground truth tie points or transformation matrix)
Creates a result file and shows the images'''


def compute_errors_homography(folder, image1_name, image2_name, matches_filename, gt_filename, threshold_correct=10):
    
    # Read images
    image1 = cv2.imread(folder + '/' + image1_name)
    image1 = image1[:, :, ::-1] #BGR to RGB

    image2 = cv2.imread(folder + '/' + image2_name)
    image2 = image2[:, :, ::-1] #BGR to RGB
    
    '''LOAD GROUND TRUTH AND CHECK ITS ACCURACY'''
    
    if gt_filename.endswith('.txt'): #GT file as tie points list
        gt_file = open(folder + '/' + gt_filename, 'r')

        pts_gt_1=[] #Arrays of points for each image
        pts_gt_2=[]
        lines = gt_file.readlines()
        for line in lines:
            coords=line.split()
            pts_gt_1.append([int(round(float(coords[0]))), int(round(float(coords[1])))])
            pts_gt_2.append([int(round(float(coords[2]))), int(round(float(coords[3])))])
            
        pts_gt_1=np.array(pts_gt_1, dtype='float32')    
        pts_gt_2=np.array(pts_gt_2, dtype='float32') 
        
        # Calculate Homography with ground truth points
        h_gt, status = cv2.findHomography(pts_gt_1, pts_gt_2)
        
        '''Error in GT'''
        res_gt = cv2.perspectiveTransform(pts_gt_1[None, :, :], h_gt); #Points of image1 transformed by calculated homography
        
        error_x_y_gt = res_gt - pts_gt_2 #Gets distances in x and y separately
        error_distance_gt = (error_x_y_gt[0,:,0]**2 + error_x_y_gt[0,:,1]**2)**0.5 #Not taking Z into account
        max_error_gt = np.max(abs(error_distance_gt))
        print('Error in Ground Truth points')
        print('Mean error: ' + str(np.mean(error_distance_gt)))
        print('Max error: ' + str(max_error_gt))
            

    elif gt_filename.endswith('.h'): #GT as matrix
        with open(folder +  + '/' + gt_file) as file_name_1:
            h_gt = np.loadtxt(file_name_1, delimiter=" ")

    #Display tranformation result for Ground Truth
    image1_transformed_gt = cv2.warpPerspective(image1, h_gt, (image2.shape[1],image2.shape[0]))
    plt.figure(figsize=[20,5])
    plt.subplot(141);plt.imshow(image1,cmap='gray');plt.title("Image 1");
    plt.subplot(142);plt.imshow(image2,cmap='gray');plt.title("Image 2");
    plt.subplot(143);plt.imshow(image1_transformed_gt,cmap='gray');plt.title("Image1 Transformed with GT");
    
    '''CHECK ERRORS OF MATCHES'''
    
    matches_file = open(folder + '/' + matches_filename, 'r')
    lines = matches_file.readlines()
    pts1=[]
    pts2=[]
    for line in lines:
        coords=line.split()
        pts1.append([int(round(float(coords[0]))), int(round(float(coords[1])))])
        pts2.append([int(round(float(coords[2]))), int(round(float(coords[3])))])
        
    pts1=np.array(pts1, dtype='float32')    
    pts2=np.array(pts2, dtype='float32') 
    
    # Calculate Homography with matches
    h_matches, status = cv2.findHomography(pts1, pts2)
    
    #Get coordinates of transformed points
    pts1_transformed = cv2.perspectiveTransform(pts1[None, :, :], h_matches)
    
    #Show images    
    image1_transformed = cv2.warpPerspective(image1, h_matches, (image2.shape[1],image2.shape[0]))
    
    plt.figure(figsize=[20,5])
    plt.subplot(141);plt.imshow(image1,cmap='gray');plt.title("Image 1");
    plt.subplot(142);plt.imshow(image2,cmap='gray');plt.title("Image 2");
    plt.subplot(143);plt.imshow(image1_transformed,cmap='gray');plt.title("Image1 Transformed with Matches");

    
    
    #Calculate error
    error_x_y = pts1_transformed - pts2 #Get differences in x and y separately
    error_distance = (error_x_y [0,:,0]**2 + error_x_y [0,:,1]**2)**0.5 #Get errors as distances between points
    max_error = np.max(abs(error_distance))
    num_pts_correct = np.count_nonzero(error_distance < threshold_correct)
    
    # Display matches
    draw_matches_error(image1, image2, pts1, pts2, error_distance,threshold_correct)
    
    print('Matches Error')
    print('Mean error: ' + str(np.mean(error_distance)))
    print('Maximum error: ' + str(max_error))

    res = open(folder + '/results_' + matches_filename[:-4] +'.txt', 'w')
    res.write('Total matches ' + str(len(error_distance)) + '\n')
    res.write('Average_error ' + str(np.mean(error_distance)) + '\n')
    res.write('Max_error ' + str(np.max(error_distance)) + '\n')
    res.write('Pts below 5 px ' + str(np.count_nonzero(error_distance < 5)) + '\n')
    res.write('Pts below 10 px ' + str(np.count_nonzero(error_distance < 10)) + '\n')
    res.write('Pts below 20 px ' + str(np.count_nonzero(error_distance < 20)) + '\n')
    res.write('Pts below 30 px ' + str(np.count_nonzero(error_distance < 30)) + '\n')
    res.close()


'''From command line'''
if __name__ == "__main__":
    folder = sys.argv[1]
    image1_name = sys.argv[2]
    image2_name = sys.argv[3]
    matches_file = sys.argv[4] 
    gt_file = sys.argv[5] #Can be a txt with tie points or .h file with transformation matrix
    threshold_correct = int(sys.argv[6]) #Theshold to consider matches correct in px, by default 10 px
    compute_errors_homography(folder, image1_name, image2_name, matches_file, gt_file, threshold_correct)
    sys.exit(0)



