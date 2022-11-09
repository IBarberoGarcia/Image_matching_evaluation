# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:34:29 2022

@author: Inés Barbero
"""

import numpy as np
import cv2
from draw_matches import draw_matches_error 
import sys
    
def get_error(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,b = img1.shape
    error=[]
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        p1 = np.asarray([x0,y0])
        p2 = np.asarray([x1,y1])
        p3 = pt1
        d = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
        error.append(d)  
    return error
    
def compute_errors_fundamental_matrix(folder, image1_name, image2_name, matches_filename, gt_filename, threshold_correct=10):
    
    # Read images
    image1 = cv2.imread(folder + '/' + image1_name)
    image1 = image1[:, :, ::-1] #BGR to RGB
    
    image2 = cv2.imread(folder + '/' + image2_name)
    image2 = image2[:, :, ::-1] #BGR to RGB
        
    '''LOAD GROUND TRUTH AND CHECK ITS ACCURACY'''
    
    if gt_filename.endswith('.txt'):
        gt_file = open(folder + '/' + gt_filename, 'r')
        lines = gt_file.readlines()
        pts_gt_1=[]
        pts_gt_2=[]
        for line in lines:
            coords=line.split()
            pts_gt_2.append([int(round(float(coords[2]))), int(round(float(coords[3])))])
            pts_gt_1.append([int(round(float(coords[0]))), int(round(float(coords[1])))])
            
        pts_gt_1=np.array(pts_gt_1, dtype='float32')    
        pts_gt_2=np.array(pts_gt_2, dtype='float32') 
        h_gt, mask = cv2.findFundamentalMat(pts_gt_1,pts_gt_2,cv2.FM_LMEDS)
        lines1 = cv2.computeCorrespondEpilines(pts_gt_2.reshape(-1,1,2), 2, h_gt)
        lines1 = lines1.reshape(-1,3)
        error_gt = get_error(image1,image2,lines1,pts_gt_1,pts_gt_2)
        
        print('Error in Ground Truth points')
        print('Mean error: ' + str(np.mean(error_gt)))
        print('Maximum error: ' + str(np.max(error_gt)))    
        
    elif gt_filename.endswith('.h'): #GT as matrix
        with open(folder + '/' + gt_filename,) as file_name_1:
            h_gt = np.loadtxt(file_name_1, delimiter=" ")

            
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
    
    # We select only inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    
        
    # Find epilines corresponding to points in right image (second image) and
    # draw its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, h_gt)
    lines1 = lines1.reshape(-1,3)
    error = get_error(image1,image2,lines1,pts1,pts2)
    draw_matches_error(image1,image2, pts1, pts2, error, threshold_correct)
    
    print('Matches Error')
    print('Mean error: ' + str(np.mean(error)))
    print('Maximum error: ' + str(np.max(error)))
    
    error=np.array(error)
    res = open(folder + '/results_' + matches_filename[:-4] +'.txt', 'w')
    res.write('Total matches ' + str(len(error)) + '\n')
    res.write('Average_error ' + str(np.mean(error)) + '\n')
    res.write('Max_error ' + str(np.max(error)) + '\n')
    res.write('Pts below 5 px ' + str(np.count_nonzero(error < 5)) + '\n')
    res.write('Pts below 10 px ' + str(np.count_nonzero(error < 10)) + '\n')
    res.write('Pts below 20 px ' + str(np.count_nonzero(error < 20)) + '\n')
    res.write('Pts below 30 px ' + str(np.count_nonzero(error < 30)) + '\n')
    res.close()
    
'''From command line'''    
if __name__ == "__main__":
    folder = sys.argv[1]
    image1_name = sys.argv[2]
    image2_name = sys.argv[3]
    matches_file = sys.argv[4] 
    gt_file = sys.argv[5] #Can be a txt with tie points or .h file with transformation matrix
    threshold_correct = int(sys.argv[6]) #Theshold to consider matches correct in px, by default 10 px
    compute_errors_fundamental_matrix(folder, image1_name, image2_name, matches_file, gt_file, threshold_correct)
    sys.exit(0)    




    
