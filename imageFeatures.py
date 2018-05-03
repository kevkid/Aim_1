#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:25:39 2018

@author: kevin
"""
import cv2
import mahotas
from skimage import feature
import numpy as np


# bins for histogram
bins = 32
class imageFeatures:
    def __init__(self):
        pass
    def fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature
    # feature-descriptor-2: Haralick Texture
    def fd_haralick(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the haralick texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick
    # feature-descriptor-3: Color Histogram
    def fd_histogram(self,image, mask=None):
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        # return the histogram
        return hist.flatten()
 
    def LocalBinaryPatterns(self, image, numPoints, radius, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(image, numPoints,
			radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
			bins=bins,
			range=(0, numPoints + 2))
 
		# normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
 
		# return the histogram of Local Binary Patterns
        return hist
    def SIFT(self, image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(image, None)
        return kp, des
    
    def HOG(self, image):
        hog = cv2.HOGDescriptor()
        h = hog.compute(image)
        return h
    
    def ORB(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Initiate STAR detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp, des = orb.detectAndCompute(image,None)        
        return kp, des
    
        