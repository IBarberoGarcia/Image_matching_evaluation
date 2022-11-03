# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:57:55 2022

@author: Inés Barbero
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_matches_error(image1, image2, pts1, pts2, error, threshold_correct):
    '''Shows image 1 and image 2 together and matches between them in red or green
    depending on error'''
    #In case image 2 is smaller
    if len(image1)>len(image2):
        #Getting the bigger side of the image
        s = len(image1)
        #Creating a dark square with NUMPY  
        new_image = np.zeros((s,len(image2[0]),3),np.uint8)
        #Pasting the 'image' in a upper part
        new_image[0:image2.shape[0],0:image2.shape[1]] = image2
        image2=new_image

    elif len(image2)>len(image1):
        #Getting the bigger side of the image
        s = len(image2)
        #Creating a dark square with NUMPY  
        f = np.zeros((s,len(image1[0]),3),np.uint8)
        #Pasting the 'image' in a upper part
        f[0:image1.shape[0],0:image1.shape[1]] = image1
        image1=f
        
    new_image = cv2.hconcat([image1,image2])
    
    i=0
    for point in pts1:
        x0 = int(point[0])
        y0 = int(point[1])
        x1 = int(pts2[i][0]+len(image1[0]))
        y1 = int(pts2[i][1])
        if error[i]>threshold_correct:
            color = [255,0,0]
        else:
            color = [0,255,0]

        new_image = cv2.line(new_image, (x0,y0), (x1,y1), color,1)
        new_image = cv2.circle(new_image,(x0,y0),5,color,-1)
        new_image = cv2.circle(new_image,(x1,y1),5,color,-1)
        i+=1
        
    #Show results
    plt.figure(figsize=(15, 15))
    plt.imshow(new_image)
    plt.show()