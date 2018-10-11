#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Fusion
Created on Mon Oct 1 14:20:57 2018

@author: kevin
"""


#%% Load dependencies
import os
if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
    os.chdir('/home/kevin/Documents/Lab/Aim_1')
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import confusion_matrix    
from sklearn import metrics
from keras.models import Model
from keras.layers import Dense, Reshape, concatenate, Lambda, Average, Maximum, Add, Multiply, Concatenate, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.optimizers import Adam
from utils import *
from keras.activations import relu

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
shape = (100,100,CHANNELS)#image shape
    
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None)
def Generate_Model(image = True, text = True, text_cnn = False):
    #set up our text input
    if image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=shape[0], 
                               name='embedding_layer', trainable=True, weights=[embedding_weights])
        text_embedding = embedding(input_text)
        #set up our image branch
        image_input = Input(shape, name='image_input')
        image_branch = get_img_branch()(image_input)
        
        #set up text branch
        text_branch = get_text_branch()(text_embedding)
        
        concat = concatenate([image_branch, text_branch])
            #Dense section
        x = Dense(1024, name='dense_layer1')(concat)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)        
        x = Dense(512, name='dense_layer2')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        output = Dense(num_class, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[input_text, image_input], outputs=[output])
    if image and not text:
        image_input = Input(shape, name='image_input')
        x = get_CNN()(image_input)
        output = Dense(num_class, activation='softmax', name='image_output')(x)
        classificationModel = Model(inputs=[image_input], outputs=[output])
    if not image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=106, 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        embedded_text = embedding(input_text)
        #text_embedding = Reshape((max_text_len,shape[0],1))(embedded_text)
        text_embedding = Lambda(embedding_3Ch)(embedded_text)
        x = get_CNN()(text_embedding)
        output = Dense(num_class, activation='softmax', name='text_output')(x)
        classificationModel = Model(inputs=[input_text], outputs=[output])
    if not image and text and text_cnn:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=106, 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        text_embedding = embedding(input_text)
        x = get_CNN_TEXT()(text_embedding)
        output = Dense(num_class, activation='softmax', name='text_output')(x)
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

#%% Train with only images
    print('Train with only images')
    image_model = Generate_Model(image=True, text=False)
    image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="model_fusion_weights/weights-improvement_image_only.hdf5"
    if LOAD_IMG_MODEL_WEIGHTS:
        print ('Loading image model weights')
        image_model.load_weights(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [reduce_lr, checkpoint, early_stopping]
    history = image_model.fit([X_image_train],[y_train], 
                             epochs=150, batch_size=128, validation_split=0.1, callbacks=callbacks_list)
#%%Test with only images
    print('Test with only images')
    y_hat = test_model(image_model, zip(X_image_test, y_test), image = True, text = False)
    
#%%train with only text:    
    print('Train with only text')
    text_model = Generate_Model(image=False, text=True, text_cnn=True)
    text_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="model_fusion_weights/weights-improvement_text_only.hdf5"
    if LOAD_TXT_MODEL_WEIGHTS:
        print ('Loading text model weights')
        text_model.load_weights(filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [reduce_lr, checkpoint, early_stopping]
    history = text_model.fit([X_text_train],[y_train], 
                             epochs=150, batch_size=128, validation_split=0.1, callbacks=callbacks_list)
#%%Test with only text
    print('Test with only text')
    y_hat = test_model(text_model, zip(X_text_test, y_test), image = False, text = True)
    
#%% Train on images + text
    print('Train on images + text')
    text_image_model = Generate_Model(image = True, text=True)
    #set optimizer:
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0005, amsgrad=True) #RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
    text_image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="model_fusion_weights/weights-improvement_image_text.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [reduce_lr, checkpoint, early_stopping]
    history = text_image_model.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=128, validation_split=0.1, callbacks=callbacks_list)
#%%test on Image + text
    print('Test on Image + text')
    y_hat = test_model(text_image_model, zip(X_text_test, X_image_test, y_test))
#%%Test with Image + Noise
    print('Testing Image + Noise')
    num_test_samples = len(test)
    text_noise = np.random.rand(num_test_samples,max_text_len)
    image_noise = np.array([np.random.rand(shape[0],shape[1]) for x in range(num_test_samples)])
    y_hat = test_model(text_image_model, zip(text_noise, X_image_test, y_test), image = True, text = True)
#%% Test with Noise + Text
    print('Testing Noise + Text')
    y_hat = test_model(text_image_model, zip(X_text_test, image_noise, y_test), image = True, text = True)
#%% Test with Noise + Noise
    print('Testing Noise + Noise')
    y_hat = test_model(text_image_model, zip(text_noise, image_noise, y_test), image = True, text = True)

#%%Merge text and image models into one big model
    outputs = [model.outputs[0] for model in [text_model, image_model]]
    output = Average()(outputs)    
    text_image_model = Model(inputs=[text_model.input, image_model.input], outputs=[output])
    text_image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    y_hat = test_model(text_image_model, zip(X_text_test, X_image_test, y_test), image = True, text = True)
    print('Test on Image + text')   
    text_image_model.evaluate(x=[X_text_test, X_image_test], y=y_test)
    print('Testing Image + Noise')
    text_image_model.evaluate(x=[text_noise, X_image_test], y=y_test)
    print('Testing Noise + Text')
    text_image_model.evaluate(x=[X_text_test, image_noise], y=y_test)
    print('Testing Noise + Noise')
    text_image_model.evaluate(x=[text_noise, image_noise], y=y_test)


#Lets try to train using these models?
    filepath="late_fusion_weights/weights-improvement_image_text.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    callbacks_list = [reduce_lr, checkpoint]
    history = text_image_model.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
    

def get_img_branch():
    Image_Branch = Sequential()
    #block 1
    Image_Branch.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    #Image_Branch.add(BatchNormalization())
    Image_Branch.add(Conv2D(64, kernel_size=(3,3), padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    Image_Branch.add(BatchNormalization())
    Image_Branch.add(Activation('relu'))
    Image_Branch.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 2
    Image_Branch.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    #Image_Branch.add(BatchNormalization())
    Image_Branch.add(Conv2D(128, kernel_size=(3,3), padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    Image_Branch.add(BatchNormalization())
    Image_Branch.add(Activation('relu'))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 3
    Image_Branch.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block3_conv1'))
    #Image_Branch.add(BatchNormalization())
    Image_Branch.add(Conv2D(256, kernel_size=(4,4), padding='valid', kernel_initializer='he_normal', name='block3_conv2'))
    Image_Branch.add(BatchNormalization())
    Image_Branch.add(Activation('relu'))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #block 4kev
    Image_Branch.add(Conv2D(512, kernel_size=(3,3), padding='valid', kernel_initializer='he_normal', name='block4_conv1'))
    #Image_Branch.add(BatchNormalization())
    Image_Branch.add(Conv2D(512, kernel_size=(4,4), padding='valid', kernel_initializer='he_normal', name='block4_conv2'))
    Image_Branch.add(BatchNormalization())
    Image_Branch.add(Activation('relu'))
    Image_Branch.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    #Image_Branch.add(Dropout(0.2))
    
    #Flatten
    Image_Branch.add(Flatten())
    return Image_Branch

def get_text_branch():
    Text_Branch = Sequential()
    #block 1
    Text_Branch.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    Text_Branch.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    Text_Branch.add(MaxPooling1D(pool_size=2, name='block1_pool'))
    Text_Branch.add(Dropout(0.1))
    #block 2
    Text_Branch.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    Text_Branch.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    Text_Branch.add(MaxPooling1D(pool_size=3, name='block2_pool'))
    Text_Branch.add(Dropout(0.1))    
    #Flatten
    Text_Branch.add(Flatten())
    return Text_Branch