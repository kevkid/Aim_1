#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:21:40 2018

@author: kevin
Image - Naive Bays
"""
import os
#if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
#    os.chdir('/home/kevin/Documents/Lab/Aim_1')
if os.getcwd() != '/home/kevin/Downloads/Aim_1':
    os.chdir('/home/kevin/Downloads/Aim_1')
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
from sklearn.model_selection import StratifiedKFold
import sklearn
from sklearn.metrics import confusion_matrix    
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
from sklearn import metrics

CLASSES = {0: 'bar', 1: 'gel', 2: 'map', 3: 'network', 4: 'plot',
         5: 'text', 6: 'box', 7: 'heatmap',8: 'medical', 9: 'nxmls', 10: 'screenshot',
         11: 'topology', 12: 'diagram', 13: 'histology', 14: 'microscopy',
         15: 'photo', 16: 'sequence', 17: 'tree', 18: 'fluorescence', 19: 'line',
         20: 'molecular', 21: 'pie', 22: 'table'}



num_mini_batches = 20
seed = 42


def train_raw_images(model, train):
    global num_mini_batches
    train_minibatch = np.array_split(train, num_mini_batches)
    #no augmentation of images
    for i, minbatch in enumerate(train_minibatch):
        print ("On minibatch {} out of {}".format(i, num_mini_batches))
        aug_minibatch = image_augmentation.no_augment_images_bulk(minbatch, shape)#.sample(frac=1)#get augmented images and shuffle them
        print("---Loaded minibatch into memory")
        model.partial_fit(aug_minibatch['x'].tolist(), aug_minibatch['y'].tolist(), list(CLASSES.keys()))
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
        model.partial_fit(aug_minibatch['x'].tolist(), aug_minibatch['y'].tolist(), list(CLASSES.keys()))
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
        '''hu = IF.fd_hu_moments(image)
        lbp = IF.LocalBinaryPatterns(image, 24, 8)
        hog = hog = np.array(IF.HOG(image)).flatten()
        global_feature = np.hstack([hist, haralick, hu, lbp, hog])#hog needs to be larger than 128 pixels
        global_feature = np.hstack([hist, haralick, hu, lbp])
        global_feature = np.hstack([haralick])'''
        global_features.append(np.hstack([hist, haralick]))
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
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed).fit(combined_features)
    return kmeans

def hist_from_BOVW(kmeans, feature_descriptors):
    print("creating histograms")
    bovw_vector = np.zeros([len(feature_descriptors), kmeans.n_clusters])#number of images x number of clusters. initiate matrix of histograms
    for index, features in enumerate(feature_descriptors):#sift descriptors in each image
        try:
            for i in kmeans.predict(features):#get label for each centroid
                bovw_vector[index, i] += 1#create individual histogram vector
        except:
            pass
    return bovw_vector
    
def k_fold_validation(clf, data, k = 5):
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    cvscores = []
    kfold_local_features = data[0]
    kfold_global_features = data[1]
    y_kfold = data[2]
    for train, test in kfold.split(X=np.zeros(len(y_kfold)), y=y_kfold):
        model = sklearn.base.clone(clf)
        print('Train K-means & Histogram Creation')
        #sift kmeans, creat our model and our histogram
        
        sift_train_kmeans = BOVW(kfold_local_features[0][train], n_clusters)
        sift_train_histogram = hist_from_BOVW(sift_train_kmeans, kfold_local_features[0][train])
        #orb kmeans, creat our model and our histogram
        orb_train_kmeans = BOVW(kfold_local_features[1][train], n_clusters)
        orb_train_histogram = hist_from_BOVW(orb_train_kmeans, kfold_local_features[1][train])
        X_train = np.concatenate((sift_train_histogram, orb_train_histogram, kfold_global_features[train]), axis=1)
        print('Test Histogram Creation')
        sift_test_histogram = hist_from_BOVW(sift_train_kmeans, kfold_local_features[0][test])
        orb_test_histogram = hist_from_BOVW(orb_train_kmeans, kfold_local_features[1][test])
        X_test = np.concatenate((sift_test_histogram, orb_test_histogram, kfold_global_features[test]), axis=1)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        y_hat = model.fit(X_train, y_kfold[train]).predict(X_test)
        cm = confusion_matrix(y_kfold[test], y_hat)
        print("Confusion Matrix ")
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(cm_df)
        scores = sum(np.equal(y_hat,np.array(y_kfold[test])))/len(y_kfold[test])
        print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
              .format(len(y_hat), sum(np.equal(y_hat,np.array(y_kfold[test]))), 
                      sum(np.equal(y_hat,np.array(y_kfold[test])))/len(y_kfold[test])))        
        cvscores.append(scores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    return model

def get_images(csv_fname = 'image_list.csv',from_db = False, classes = list(range(0,23)), uniform = True):
    if from_db:
        image_list = get_images_from_db.get_images_from_db("labeled")
        image_list.to_csv(csv_fname)
    else:
        image_list = pd.read_csv(csv_fname)
    #get classes from input
    image_list_subset = image_list[image_list.class_id.isin(classes)]
    if uniform:
        image_list_subset = image_list_subset.groupby('class_id').head(
                min(image_list_subset.class_id.value_counts()))
    return image_list_subset
'''
Gives us our train and test data. In doing so it extracts the features, creates
the BOVW, and our histograms for the local data. Furthuremore it concatenates
the local and global feature descriptors and scales all of them
'''
def get_train_test(data, test_size = 0.2, scale = True):
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    '''Train'''
    print('Getting training set')
    train_local_features, train_global_features, y_train = extract_features(train)
    #sift kmeans, create our model and our histogram
    sift_train_kmeans = BOVW(train_local_features[0], n_clusters)
    sift_train_histogram = hist_from_BOVW(sift_train_kmeans, train_local_features[0])
    #orb kmeans, creat our model and our histogram
    orb_train_kmeans = BOVW(train_local_features[1], n_clusters)
    orb_train_histogram = hist_from_BOVW(orb_train_kmeans, train_local_features[1])
    x_train = np.concatenate((sift_train_histogram, orb_train_histogram, train_global_features), axis=1)
    
    '''Test'''
    print('Getting testing set')
    test_local_features, test_global_features, y_test = extract_features(test)
    sift_test_histogram = hist_from_BOVW(sift_train_kmeans, test_local_features[0])
    orb_test_histogram = hist_from_BOVW(orb_train_kmeans, test_local_features[1])
    x_test = np.concatenate((sift_test_histogram, orb_test_histogram, test_global_features), axis=1)
    
    '''Scale Data'''
    if scale:
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
    return x_train, y_train, x_test, y_test

#shape = image_augmentation.get_average_shape(image_list)#Width x Height
if __name__ == '__main__':
    #initial Parameters
    n_clusters = 100#clusters for BOVW
    shape = (106,106)#image shape
    class_names = ['bar', 'gel', 'network', 'plot', 'histology', 'sequence',
                           'line', 'molecular']
    image_classes = [list(CLASSES.keys())[list(CLASSES.values()).index(x)] for 
                     x in class_names]
    #get images
    images = get_images(csv_fname='image_list.csv', classes=image_classes)
    #get train and Test set
    x_train, y_train, x_test, y_test = get_train_test(images)
    
    #kfold
    kfold_local_features, kfold_global_features, y_kfold = extract_features(images)
    kfold_local_features = np.array(kfold_local_features)
    kfold_global_features = np.array(kfold_global_features)
    y_kfold = np.array(y_kfold)
    kfold_data = [kfold_local_features, kfold_global_features, y_kfold]
    del kfold_local_features, kfold_global_features, y_kfold
    
    
    '''Naive Bays'''
    model = GaussianNB()
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    cm = confusion_matrix(y_test, y_hat)
    print("Confusion Matrix for: Naive Bays")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    #k-fold Validation:
    model = GaussianNB()
    k_fold_validation(model, kfold_data, 5)
    
    '''Decision Tree'''
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=seed)
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    cm = confusion_matrix(y_test, y_hat)
    print("Confusion Matrix for: Decision Tree")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    model = DecisionTreeClassifier(random_state=seed)
    k_fold_validation(model, kfold_data, 5)
    
    '''Random Forest, just here for testing'''
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    cm = confusion_matrix(y_test, y_hat)
    print("Confusion Matrix for: Random Forest")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    k_fold_validation(model, kfold_data, 5)
    
    '''SVM'''
    from sklearn.svm import SVC
    model = SVC()
    y_hat = model.fit(x_train, y_train).predict(x_test)
    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    cm = confusion_matrix(y_test, y_hat)
    print("Confusion Matrix for: SVM")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    model = SVC()
    k_fold_validation(model, kfold_data, 5)
    
    '''Neural networks'''
    def get_images_from_df(df):
        images = []
        for i, (index, sample) in enumerate(df.iterrows()):
                #read image
                image = cv2.imread(sample["location"])
                image = cv2.resize(image, shape)
                if len(image.shape) < 3:#if there is no channel data
                    image = np.stack((image,)*3, -1)
                images.append(image)
        return np.array(images)
    NN_Class_Names = {0:'bar',1:'gel',2:'network',3:'plot',4:'histology',5:'sequence',6:'line',7:'molecular'}
    train, test = train_test_split(images, test_size=0.2, random_state=seed)
    X_train = get_images_from_df(train)/255
    X_test = get_images_from_df(test)/255
    y_train = np.array([class_id for class_id in train.loc[: ,'class_id']]).reshape((-1, 1))
    y_test = np.array([class_id for class_id in test.loc[: ,'class_id']]).reshape((-1, 1))
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    onehotencoder_train = OneHotEncoder()
    onehotencoder_test = OneHotEncoder()
    y_train = onehotencoder_train.fit_transform(y_train).toarray()
    y_test = onehotencoder_test.fit_transform(y_test).toarray()
    
    
    '''Conv Net'''
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    def get_CNN():
        CNN_Model = Sequential()
        #block 1
        CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', input_shape=(106, 106, 3), name='block1_conv1'))
        CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
        CNN_Model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
        CNN_Model.add(Dropout(0.2))
        
        #block 2
        CNN_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
        CNN_Model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
        CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
        CNN_Model.add(Dropout(0.2))
        
        #block 3
        CNN_Model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv1'))
        CNN_Model.add(Conv2D(256, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv2'))
        CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
        CNN_Model.add(Dropout(0.2))
        
        #block 4
        CNN_Model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv1'))
        CNN_Model.add(Conv2D(512, kernel_size=(4,4), activation='relu', padding='valid', kernel_initializer='he_normal', name='block4_conv2'))
        CNN_Model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
        CNN_Model.add(Dropout(0.5))
        
        #Flatten
        CNN_Model.add(Flatten())
        #Dense section
        CNN_Model.add(Dense(512, activation='relu', name='dense_layer1'))
        CNN_Model.add(Dropout(0.5))
        CNN_Model.add(Dense(512, activation='relu', name='dense_layer2'))
        CNN_Model.add(Dropout(0.5))
        CNN_Model.add(Dense(8, activation='softmax', name='final_layer'))
        return CNN_Model
    
    model = get_CNN()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('model_weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train, batch_size=128,
                        epochs=200, verbose=1, validation_split=0.10, callbacks=callbacks_list)
    model.load_weights('model_weights.hdf5')
    '''ValueError: In case of mismatch between the provided input data and the 
    model's expectations, or in case a stateful model receives a number of samples that is not a multiple of the batch size.'''
    y_hat = model.predict(X_test)
    #convert to categorical from one-hot
    y_hat = [ np.argmax(t) for t in y_hat ]
    y_test = [ np.argmax(t) for t in y_test ]
    test_wrong = [im for im in zip(X_test, y_hat,y_test) if im[1] != im[2]]#Need this for images

    print("Number of correctly labeled points out of a total {} points : {}. An accuracy of {}"
          .format(len(y_hat), sum(np.equal(y_hat,np.array(y_test))), 
                  sum(np.equal(y_hat,np.array(y_test)))/len(y_hat)))
    cm = confusion_matrix(y_test, y_hat)
    y_test = np.array([class_id for class_id in test.loc[: ,'class_id']]).reshape((-1, 1))
    y_test = onehotencoder_test.fit_transform(y_test).toarray()#reset back to normal for prediction
    print("Confusion Matrix for: CNN")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    print('blue = ground truth, red = prediction')
    plt.figure(figsize=(20, 20), dpi=300)
    for ind, val in enumerate(test_wrong[:118]):
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.subplot(10, 12, ind + 1)
        im = val[0]
        plt.axis("off")
        plt.text(0, 0, NN_Class_Names[val[2]], fontsize=14, color='blue')
        plt.text(60, 0, NN_Class_Names[val[1]], fontsize=14, color='red')
        plt.imshow(im, cmap='gray')
    

    '''NN K-Fold'''
    #Get training data
    X_kfold = get_images_from_df(images)/255
    y_kfold = np.array([class_id for class_id in images.loc[: ,'class_id']]).reshape((-1, 1))
    #one-hot encode labels
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder_kfold = OneHotEncoder()
    onehotencoder_kfold.fit(y_kfold)
    #set K folds to 5
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    count = 1
    for train, test in kfold.split(X_kfold, y_kfold):
        random.shuffle(train)
        model = get_CNN()
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        weight_str = 'model_weights_kfold_' + str(count) + '.hdf5'
        checkpoint = ModelCheckpoint(weight_str, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        
        '''Have to make the y_train onehot encoded otherwise it wont predict properly'''
        '''When using validation_split = 0.10 it takes last 10% of data, so if 
        your kfold does not shuffle it well enough it will give low validation 
        accuracy, and in this case it wont even train class 16 well. Class 16
        is the last class in the list. By coincidence, we have 314 images per 
        class in our training set. When we pull the last 10 percent of the images
        for validation, it is also 314 images, exactly missing one class. This
        is why we get such a low validation score, it does not see one class and
        cant really train that well because of it. It improves a little, but is crap.
        Keras Documentation: validation_split: Float between 0 and 1. Fraction 
        of the training data to be used as validation data. The model will set 
        apart this fraction of the training data, will not train on it, and will 
        evaluate the loss and any model metrics on this data at the end of each 
        epoch. The validation data is selected from the last samples in the 
        x and y data provided, before shuffling.
        Performing random.shuffle(train) solves the problem
        '''
        history = model.fit(X_kfold[train], onehotencoder_kfold.transform(y_kfold).toarray()[train], batch_size=128,
                        epochs=200, verbose=1, validation_split=0.10, callbacks=callbacks_list)
        # evaluate the model
        model.load_weights(weight_str)
        scores = model.evaluate(X_kfold[test], onehotencoder_kfold.transform(y_kfold).toarray()[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        count += 1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    metrics_y_test = [ np.argmax(t) for t in onehotencoder_kfold.transform(y_kfold).toarray()[test]]
    metrics_y_hat  = [ np.argmax(t) for t in model.predict(X_kfold[test]) ]
    report = metrics.classification_report(metrics_y_test, metrics_y_hat, target_names=list(NN_Class_Names.values()))
    print(report)
    print("Confusion Matrix for: CNN")
    cm = confusion_matrix(metrics_y_test, metrics_y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    test_wrong = [im for im in zip(X_kfold[test], metrics_y_hat,metrics_y_test) if im[1] != im[2]]#Need this for images
    print('blue = ground truth, red = prediction')
    plt.figure(figsize=(12, 12), dpi=300)
    for ind, val in enumerate(test_wrong[:50]):
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.subplot(10, 10, ind + 1)
        im = val[0]
        plt.axis("off")
        plt.text(0, 0, NN_Class_Names[val[2]], fontsize=8, color='blue')
        plt.text(60, 0, NN_Class_Names[val[1]], fontsize=8, color='red')
        plt.imshow(im, cmap='gray')

