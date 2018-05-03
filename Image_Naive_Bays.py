#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:21:40 2018

@author: kevin
Image - Naive Bays
"""
import os
if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
    os.chdir('/home/kevin/Documents/Lab/Aim_1')
from sklearn.naive_bayes import GaussianNB
import get_images_from_db
import image_augmentation
from random import randint
import imageio
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from imageFeatures import imageFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
CLASSES = {'bar': 0, 'gel': 1, 'map': 2, 'network': 3,  'plot': 4, 'text': 5, 
           'box': 6, 'heatmap': 7, 'medical': 8, 'nxmls': 9, 'screenshot': 10, 
           'topology': 11, 'diagram': 12, 'histology': 13, 'microscopy': 14, 'photo': 15, 
           'sequence': 16, 'tree': 17, 'fluorescence': 18, 'line': 19, 'molecular': 20, 
           'pie': 21, 'table': 22}
num_mini_batches = 20


def train_raw_images(model, train):
    global num_mini_batches
    train_minibatch = np.array_split(train, num_mini_batches)
    #no augmentation of images
    for i, minbatch in enumerate(train_minibatch):
        print ("On minibatch {} out of {}".format(i, num_mini_batches))
        aug_minibatch = image_augmentation.no_augment_images_bulk(minbatch, shape)#.sample(frac=1)#get augmented images and shuffle them
        print("---Loaded minibatch into memory")
        model.partial_fit(aug_minibatch['x'].tolist(), aug_minibatch['y'].tolist(), list(CLASSES.values()))
        print("---Fitting Model")
        aug_minibatch = None
        del aug_minibatch
    return model

def train_raw_images_augmentation(model, train):
    global num_mini_batches
    train_minibatch = np.array_split(train, num_mini_batches)
    #augmentation of images
    for i, minbatch in enumerate(train_minibatch):
        print ("On minibatch {} out of {}".format(i, num_mini_batches))
        aug_minibatch = image_augmentation.augment_images_bulk(minbatch, shape)#.sample(frac=1)#get augmented images and shuffle them
        print("---Loaded minibatch into memory")
        model.partial_fit(aug_minibatch['x'].tolist(), aug_minibatch['y'].tolist(), list(CLASSES.values()))
        print("---Fitting Model")
        aug_minibatch = None
        del aug_minibatch
    return model

def extract_features(df):
    IF = imageFeatures()
    global_features = []
    sift_features = []
    orb_features = []
    labels = []
    for i, (index, sample) in enumerate(df.iterrows()):
        #read image
        image = cv2.imread(sample["location"])
        image = cv2.resize(image, shape)
        #global features
        hist = IF.fd_histogram(image)
        haralick = IF.fd_haralick(image)
        hu = IF.fd_hu_moments(image)
        lbp = IF.LocalBinaryPatterns(image, 24, 8)
        hog = hog = np.array(IF.HOG(image)).flatten()
        global_feature = np.hstack([hist, haralick, hu, lbp, hog])
        global_features.append(global_feature)
        #local features
        kp, des = IF.SIFT(image)
        if len(kp) == 0:
            des = np.zeros(128)
        sift_features.append(des)
        
        kp, des = IF.ORB(image)
        if len(kp) == 0:
            des = np.zeros(32)
        orb_features.append(des)
        
        labels.append(sample["class_id"])
    #normalize global features
    global_features = normalize(global_features)
    local_features = [sift_features, orb_features]
    return local_features, global_features, labels

def BOVW(feature_descriptors, n_clusters = 100):
    print("Bag of visual words with {} clusters".format(n_clusters))
    #take all features and put it into a giant list
    combined_features = np.vstack(np.array(feature_descriptors))
    #train kmeans on giant list
    print("Starting K-means training")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(combined_features)
    print("Finished K-means training, moving on to prediction")
    bovw_vector = np.zeros([len(feature_descriptors), n_clusters])#number of images x number of clusters. initiate matrix of histograms
    for index, features in enumerate(feature_descriptors):#sift descriptors in each image
        try:
            for i in kmeans.predict(features):#get label for each centroid
                bovw_vector[index, i] += 1#create individual histogram vector
        except:
            pass
    return bovw_vector#this should be our histogram

if __name__ == '__main__':
    n_clusters = 1000
    #set model
    model = GaussianNB()
    #image_list = get_images_from_db.get_images_from_db("labeled")
    #image_list.to_csv("image_list.csv")
    image_list = pd.read_csv("image_list.csv")
    image_list_subset = image_list.groupby('class_id').head(5)#image_list.loc[(image_list["class_id"] == 0) | (image_list["class_id"] == 19)]
    #shape = image_augmentation.get_average_shape(image_list)#Width x Height
    shape = (330,230)
    #shape = tuple([x//2 for x in shape])
    #random_images = [randint(0, len(image_list)) for i in range(40)]
    #image_list_subset = image_list.iloc[random_images]
    #split image list so we dont mix up the augmented images
    #train, test = train_test_split(image_list_subset, test_size=0.1, random_state=42)
    
    
    train, test = train_test_split(image_list_subset, test_size=0.1, random_state=42)
    
    train_local_features, train_global_features, y_train = extract_features(train)
    sift_train_histogram = BOVW(train_local_features[0], n_clusters)
    orb_train_histogram = BOVW(train_local_features[1], n_clusters)
    
    x_train = np.concatenate((normalize(sift_train_histogram), normalize(orb_train_histogram), train_global_features), axis=1)
    import matplotlib.pyplot as plt
    plt.plot(train[10], 'o')
    plt.ylabel('frequency');
    plt.xlabel('features');
    
    test_local_features, test_global_features, y_test = extract_features(test)
    sift_test_histogram = BOVW(test_local_features[0], n_clusters)
    orb_test_histogram = BOVW(test_local_features[1], n_clusters)
    x_test = np.concatenate((normalize(sift_test_histogram), normalize(orb_test_histogram), test_global_features), axis=1)
    
    import heapq, random
    
    '''Naive Bays'''
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    
    '''Decision Tree'''
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=9)
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    
    '''Random Forest, just here for testing'''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=9)
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    '''SVM'''
    from sklearn.svm import SVC
    model = SVC()
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))



































'''Test'''
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    data = unpickle('data_batch_1')
    images = data[b'data']
    labels = data[b'labels']
    new_images = np.zeros((10000, 32,32,3))
    image_features = []
    for i, img in enumerate(images):
        new_images[i] = np.reshape(img, (32,32,3))
    
    for image in new_images:
        kp, des = IF.SIFT(np.uint8(image))
        image_features.append(des)
    n_clusters = 100
    image_hist = BOVW(image_features[:500], n_clusters)
    
    
    
    
    test_data = unpickle('test_batch')
    test_images = test_data[b'data']
    test_labels = test_data[b'labels']
    test_new_images = np.zeros((10000, 32,32,3))
    test_image_features = []
    for i, img in enumerate(test_images):
        test_new_images[i] = np.reshape(img, (32,32,3))
    
    for image in test_new_images:
        kp, des = IF.SIFT(np.uint8(image))
        test_image_features.append(des)
    n_clusters = 100
    test_image_hist = BOVW(image_features[:500], n_clusters)
    
    y_hat = model.fit(image_hist, labels[:500]).predict(test_image_hist)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(test_labels[:500]))), 
                  sum(np.equal(y_hat,np.array(test_labels[:500])))/len(y_hat)))
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=9)
    y_hat = model.fit(image_hist, labels[:500]).predict(test_image_hist)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(test_labels[:500]))), 
                  sum(np.equal(y_hat,np.array(test_labels[:500])))/len(y_hat)))
    
    IF = imageFeatures()
    image = cv2.imread(test.iloc[1]['location'])
    image = cv2.resize(image, shape)
    kp, des = IF.ORB(image)
    kp, sift = IF.SIFT(image)
    hog = [item for sublist in IF.HOG(image) for item in sublist] 
    hog = np.array(IF.HOG(image)).flatten()
    sift
    haralick = IF.fd_haralick(image)
    hu = IF.fd_hu_moments(image)
    hist = IF.fd_histogram(image)
    sift
    rescaled = np.hstack([sift.flatten(), hist])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(rescaled)
    
    from sklearn import datasets

    iris = datasets.load_iris()
####Testing    
    test = image_augmentation.augment_images_bulk(test.iloc[:20], shape)
    
    #model.fit(X, y);
    #X_train, X_test, y_train, y_test = train_test_split(array['x'].tolist(), array['y'].tolist(), test_size=0.1, random_state=42)
    #y_pred = model.fit(train['x'].tolist(), train['y'].tolist()).predict(test['x'].tolist())
    y_pred = model.predict(test['x'].tolist())
    print("Number of mislabeled points out of a total {} points : {}. An accuracy of {}".format(len(test), sum(y_pred!=test['y'].tolist()), 1-sum(y_pred!=test['y'].tolist())/len(test)))
        

'''
import imageio
image = imageio.imread(image_list.iloc[4]["location"])
displayImage(image)
if len(image.shape) < 3:#if there is no channel data
        image = np.stack((image,)*3, -1)
displayImage(image)


import matplotlib.pyplot as plt
def displayImage(image):
    plt.imshow(image)
    plt.show()
displayImage(array[9][12])

'''