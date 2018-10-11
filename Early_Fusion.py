#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 23:12:23 2018

@author: kevin

Fusion model should look like 2 inputs being text and image

"""
#%% Load dependencies
import os
if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
    os.chdir('/home/kevin/Documents/Lab/Aim_1')
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import confusion_matrix    
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Dense, Reshape, concatenate, Lambda
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.optimizers import Adam
from utils import *

SRC = 'PMC'

if SRC=='PMC':
    CLASSES = {0: 'bar', 1: 'gel', 2: 'map', 3: 'network', 4: 'plot',
             5: 'text', 6: 'box', 7: 'heatmap',8: 'medical', 9: 'nxmls', 10: 'screenshot',
             11: 'topology', 12: 'diagram', 13: 'histology', 14: 'microscopy',
             15: 'photo', 16: 'sequence', 17: 'tree', 18: 'fluorescence', 19: 'line',
             20: 'molecular', 21: 'pie', 22: 'table'}
    
    class_names = ['bar', 'gel', 'network', 'plot', 'histology', 'sequence',
                               'line', 'molecular']
else:
    class_names = ['cat','dog']
num_class = len(class_names)
NN_Class_Names = dict(enumerate(class_names))
#Set parameters
create_word_vec = False
w2v_epochs = 10
seed = 0
max_text_len = 200
vocabulary_size = 10000
csv_fname = 'image_list.csv'
coco_loc = '/home/kevin/Documents/Lab/coco_dataset'
#coco_loc = '/media/kevin/1142-5B72/coco_dataset'
filters = ''
use_glove = False
LOAD_IMG_MODEL_WEIGHTS = False
LOAD_TXT_MODEL_WEIGHTS = False
CHANNELS = 3#should be either 3 or 1 for the number of channels
shape = (106,106,CHANNELS)#image shape
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None)
def Generate_Model(image = True, text = True):
    #set up our text input
    if image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=shape[0], 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        embedded_text = embedding(input_text)
        text_embedding = Lambda(embedding_3Ch)(embedded_text)
        #set up our image input
        image_input = Input(shape, name='image_input')
        '''merge our data for early fusion. Essentially we are just concatenating our data together'''
        merged = concatenate([text_embedding, image_input], axis=1, name='merged')
        x = get_CNN()(merged)
        output = Dense(num_class, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[input_text, image_input], outputs=[output])
    if image and not text:
        image_input = Input(shape, name='image_input')
        x = get_CNN()(image_input)
        output = Dense(num_class, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[image_input], outputs=[output])
    if not image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=106, 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        embedded_text = embedding(input_text)
        text_embedding = Lambda(embedding_3Ch)(embedded_text)
        x = get_CNN()(text_embedding)
        output = Dense(num_class, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[input_text], outputs=[output])
    return classificationModel



#%% Section 0: Get data
if __name__ == '__main__':
    print('Getting Data')
    if SRC == 'PMC':
        image_classes = [list(CLASSES.keys())[list(CLASSES.values()).index(x)] for 
                     x in class_names]  
        data = load_PMC(csv_fname, image_classes, uniform=True)
    else:
        data = load_COCO(coco_loc, class_names=class_names)
#%%Section 0.5 create embeddings - Optional
    if create_word_vec:
        print('Section 0.5: create embeddings')
        from w2v_keras import w2v_keras
        w2v = w2v_keras(vocabulary_size=vocabulary_size, window_size=3, filters='')
        w2v.fit(data['caption'], epochs=w2v_epochs)
#%% Section 1: Generate our sequences from text
    print('Get model data')
    (X_text_train, X_text_test, X_image_train, X_image_test, y_train, y_test, 
     tokenizer) = get_model_data(data, seed=seed, test_size = 0.2, vocabulary_size = vocabulary_size, 
              filters = filters, max_text_len=max_text_len, shape=shape)
#%% Section 1.5: Load word embeddings
    if use_glove:
        embedding_weights = load_glove('/home/kevin/Downloads', 
                            take(vocabulary_size, tokenizer.word_index.items()))
    elif create_word_vec:#get weights from w2v model directly if trained.
        embedding_weights = mapTokensToEmbedding(w2v.get_embeddings(), 
                                                 tokenizer.word_index, vocabulary_size)
    else:
        #load embedding weights from file
        location = 'w2v_embeddings.json'
        import json
        with open(location) as f:
            embs = json.load(f)
            emb = json.loads(embs)
        embedding_weights = mapTokensToEmbedding(emb, tokenizer.word_index, 
                                                 vocabulary_size)
#%% Train on images + text
    print('Train on images + text')
    classificationModel = Generate_Model(image = True, text=True)
    #set optimizer:
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0005, amsgrad=True) #RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
    classificationModel.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="early_fusion_weights/weights-improvement_image_text.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    callbacks_list = [reduce_lr, checkpoint, early_stopping]
    history = classificationModel.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
#%%test on Image + text
    print('Test on Image + text')
    test_model(classificationModel, zip(X_text_test, X_image_test, y_test), image = True, text = True)
#%%Test with Image + Noise
    print('Testing Image + Noise')
    num_test_samples = len(y_test)
    text_noise = np.random.rand(num_test_samples,max_text_len)
    image_noise = np.array([np.random.rand(shape[0],shape[1]) for x in range(num_test_samples)])
    test_model(classificationModel, zip(text_noise, X_image_test, y_test), image = True, text = True)
#%% Test with Noise + Text
    print('Testing Noise + Text')
    test_model(classificationModel, zip(X_text_test, image_noise, y_test), image = True, text = True)
#%% Test with Noise + Noise
    print('Testing Noise + Noise')
    test_model(classificationModel, zip(text_noise, image_noise, y_test), image = True, text = True)
#%% Train with only images
    print('Train with only images')
    classificationModel = Generate_Model(image=True, text=False)
    classificationModel.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="early_fusion_weights/weights-improvement_image_only.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [reduce_lr, checkpoint, early_stopping]
    history = classificationModel.fit([X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
#%%Test with only images
    print('Test with only images')
    test_model(classificationModel, zip(X_image_test, y_test), image = True, text = False)
    
#%%train with only text:    
    print('Train with only text')
    classificationModel = Generate_Model(image=False, text=True)
    classificationModel.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="early_fusion_weights/weights-improvement_text_only.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [reduce_lr, checkpoint, early_stopping]
    history = classificationModel.fit([X_text_train],[y_train], 
                             epochs=150, batch_size=32, validation_split=0.1, callbacks=callbacks_list)
#%%Test with only text
    print('Test with only text')
    test_model(classificationModel, zip(X_text_test, y_test), image = False, text = True)
    