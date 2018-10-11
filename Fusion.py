#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:30:34 2018

@author: kevin
"""
import os
if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
    os.chdir('/home/kevin/Documents/Lab/Aim_1')
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import confusion_matrix    
from sklearn import metrics
from keras.models import Model
from keras.layers import Dense, Reshape, concatenate, Lambda, Average, Maximum, Add, Multiply
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.optimizers import Adam
from utils import *

class Fusion():
    def __init__(self, Data = None, params = None):
        if data is None:
            print('You need to pass in data in the form of a pandas dataframe with the following ' 
                  'columns:\n\nclass_id\nlocation\ncaption')
            return -1
        if params is None:
            print('You must pass in a dictionary of parameters which have the '
                  'following keys:\nclass_names\ncreate_word_vec\nw2v_epochs\n'
                  'seed\nmax_text_len\nvocabulary_size\nfilters\nwindow_size\n'
                  'embedding_location\nCHANNELS\nresize_shape')

        self.num_class = len(params['class_names'])
        self.NN_Class_Names = dict(enumerate(params['class_names']))
        self.create_word_vec = params['create_word_vec']
        self.w2v_epochs = params['w2v_epochs']
        self.seed = params['seed']
        self.max_text_len = params['max_text_len']
        self.vocabulary_size = params['vocabulary_size']
        self.filters = params['filters']
        self.window_size = params['window_size']
        self.embedding_location = params['embedding_location']#Should be 1 of the 3: glove, memory, disk
        self.shape = params['resize_shape'] + (params['CHANNELS'],)
        self.embedding_path = params['embedding_path']
        if self.create_word_vec:
            print('Section 0.5: create embeddings')
            from w2v_keras import w2v_keras
            w2v = w2v_keras(vocabulary_size=self.vocabulary_size, window_size=self.window_size, 
                            filters=self.filters, vector_dim=self.shape[0])
            w2v.fit(data['caption'], epochs=self.w2v_epochs)
            w2v.save_embeddings(params['embedding_path'])
        print('Get model data')
        (self.X_text_train, self.X_text_test, self.X_image_train, self.X_image_test, self.y_train, self.y_test, 
         tokenizer) = get_model_data(data, seed=self.seed, test_size = 0.2, vocabulary_size = self.vocabulary_size, 
                  filters = self.filters, max_text_len=self.max_text_len, shape=self.shape)
        if self.embedding_location == 'glove':
            embedding_weights = load_glove('/home/kevin/Downloads', 
                                take(self.vocabulary_size, tokenizer.word_index.items()))
        elif self.embedding_location == 'memory':#get weights from w2v model directly if trained.
            embedding_weights = mapTokensToEmbedding(w2v.get_embeddings(), 
                                                     tokenizer.word_index, self.vocabulary_size)
        else:#load embedding weights from file
            location = self.embedding_path
            import json
            with open(location) as f:
                embs = json.load(f)
                emb = json.loads(embs)
            embedding_weights = mapTokensToEmbedding(emb, tokenizer.word_index, 
                                                     self.vocabulary_size)
        self.embedding_weights = embedding_weights

    def get_model_inputs(self):
        return (self.X_text_train, self.X_text_test, self.X_image_train, self.X_image_test, self.y_train, self.y_test)
        
    
class Late_Fusion(Fusion):
    def Generate_Model(self,image = True, text = True, text_cnn = False):
        #set up our text input
        if image and text:
            input_text = Input(shape=(self.max_text_len,), name='text_input')
            embedding = Embedding(input_length=self.max_text_len, input_dim=self.vocabulary_size, output_dim=self.shape[0], 
                                   name='embedding_layer', trainable=False, weights=[self.embedding_weights])
            text_embedding = embedding(input_text)
            #set up our image branch
            image_input = Input(self.shape, name='image_input')
            image_branch = get_CNN()(image_input)
            image_branch_out = Dense(self.num_class, activation='softmax', name='image_branch_output')(image_branch)
            image_model = Model(inputs=[image_input], outputs=[image_branch_out])
            #set up text branch
            text_branch = get_CNN_TEXT()(text_embedding)
            text_branch_out = Dense(self.num_class, activation='softmax', name='text_branch_output')(text_branch)
            text_model = Model(inputs=[input_text], outputs=[text_branch_out])
            outputs = [model.outputs[0] for model in [text_model, image_model]]
            output = Average()(outputs)
            classificationModel = Model(inputs=[text_model.input, image_model.input], outputs=[output])
        if image and not text:
            image_input = Input(self.shape, name='image_input')
            x = get_CNN()(image_input)
            output = Dense(self.num_class, activation='softmax', name='image_output')(x)
            classificationModel = Model(inputs=[image_input], outputs=[output])
        if not image and text:
            input_text = Input(shape=(self.max_text_len,), name='text_input')
            embedding = Embedding(input_length=self.max_text_len, input_dim=self.vocabulary_size, output_dim=self.shape[0], 
                                   name='embedding_layer', trainable=False, weights=[self.embedding_weights])
            embedded_text = embedding(input_text)
            #text_embedding = Reshape((max_text_len,shape[0],1))(embedded_text)
            text_embedding = Lambda(embedding_3Ch)(embedded_text)
            x = get_CNN()(text_embedding)
            output = Dense(self.num_class, activation='softmax', name='text_output')(x)
            classificationModel = Model(inputs=[input_text], outputs=[output])
        if not image and text and text_cnn:
            input_text = Input(shape=(self.max_text_len,), name='text_input')
            embedding = Embedding(input_length=self.max_text_len, input_dim=self.vocabulary_size, output_dim=self.shape[0], 
                                   name='embedding_layer', trainable=False, weights=[self.embedding_weights])
            text_embedding = embedding(input_text)
            x = get_CNN_TEXT()(text_embedding)
            output = Dense(self.num_class, activation='softmax', name='text_output')(x)
            classificationModel = Model(inputs=[input_text], outputs=[output])
        return classificationModel

class Early_Fusion(Fusion):
    def Generate_Model(self, image = True, text = True):
    #set up our text input
        if image and text:
            input_text = Input(shape=(self.max_text_len,), name='text_input')
            embedding = Embedding(input_length=self.max_text_len, input_dim=vocabulary_size, output_dim=self.shape[0], 
                                   name='embedding_layer', trainable=False, weights=[self.embedding_weights])
            embedded_text = embedding(input_text)
            text_embedding = Lambda(embedding_3Ch)(embedded_text)
            #set up our image input
            image_input = Input(self.shape, name='image_input')
            '''merge our data for early fusion. Essentially we are just concatenating our data together'''
            merged = concatenate([text_embedding, image_input], axis=1, name='merged')
            x = get_CNN()(merged)
            output = Dense(self.num_class, activation='softmax', name='output')(x)
            classificationModel = Model(inputs=[input_text, image_input], outputs=[output])
        if image and not text:
            image_input = Input(self.shape, name='image_input')
            x = get_CNN()(image_input)
            output = Dense(self.num_class, activation='softmax', name='output')(x)
            classificationModel = Model(inputs=[image_input], outputs=[output])
        if not image and text:
            input_text = Input(shape=(self.max_text_len,), name='text_input')
            embedding = Embedding(input_length=self.max_text_len, input_dim=self.vocabulary_size, output_dim=self.shape[0], 
                                   name='embedding_layer', trainable=False, weights=[self.embedding_weights])
            embedded_text = embedding(input_text)
            text_embedding = Lambda(embedding_3Ch)(embedded_text)
            x = get_CNN()(text_embedding)
            output = Dense(self.num_class, activation='softmax', name='output')(x)
            classificationModel = Model(inputs=[input_text], outputs=[output])
        return classificationModel
    
#%% Main code
if __name__ == '__main__':
    fusion_type = 'late'
    csv_fname = 'image_list.csv'
    coco_loc = '/home/kevin/Documents/Lab/coco_dataset'
    #coco_loc = '/media/kevin/1142-5B72/coco_dataset'
    SRC = 'COCO'
    LOAD_IMG_MODEL_WEIGHTS = True
    LOAD_TXT_MODEL_WEIGHTS = False
    if SRC=='PMC':
        CLASSES = {0: 'bar', 1: 'gel', 2: 'map', 3: 'network', 4: 'plot',
                 5: 'text', 6: 'box', 7: 'heatmap',8: 'medical', 9: 'nxmls', 10: 'screenshot',
                 11: 'topology', 12: 'diagram', 13: 'histology', 14: 'microscopy',
                 15: 'photo', 16: 'sequence', 17: 'tree', 18: 'fluorescence', 19: 'line',
                 20: 'molecular', 21: 'pie', 22: 'table'}
        
        class_names = ['bar', 'gel', 'network', 'plot', 'histology', 'sequence',
                                   'line', 'molecular']
    else:
        class_names = ['cat','dog','bird','horse','cow','elephant','zebra','giraffe']

    print('Getting Data')
    if SRC == 'PMC':
        image_classes = [list(CLASSES.keys())[list(CLASSES.values()).index(x)] for 
                     x in class_names]  
        data = load_PMC(csv_fname, image_classes, uniform=True)
    else:
        data = load_COCO(coco_loc, class_names=class_names)
#TODO Make sure we can detect if we create a w2v model we load it and we dont use glove/vice versa
    params = {'class_names': class_names, 'create_word_vec': True, 
              'w2v_epochs': 10, 'seed': 0, 'max_text_len': 200, 'vocabulary_size': 10000,
              'filters':'', 'window_size':3, 'embedding_location': 'disk', 'CHANNELS':3, 
              'resize_shape':(100,100), 'embedding_path':'w2v_embeddings.json'}
    if params['create_word_vec']:
        params['embedding_location'] = 'memory'
    else:
        params['embedding_location'] = 'disk'
        
    if fusion_type == 'late':
        fusion = Late_Fusion(data, params)
        folder = 'late_fusion_weights'
    else:
        fusion = Early_Fusion(data, params)
        folder = 'early_fusion_weights'
    
    X_text_train, X_text_test, X_image_train, X_image_test, y_train, y_test = fusion.get_model_inputs()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None)
#%% Train with only images
    print('Train with only images')
    image_model = fusion.Generate_Model(image=True, text=False)
    image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="{}/weights-improvement_image_only.hdf5".format(folder)
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
    text_model = fusion.Generate_Model(image=False, text=True, text_cnn=True)
    text_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="{}/weights-improvement_text_only.hdf5".format(folder)
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
    text_image_model = fusion.Generate_Model(image = True, text=True)
    #set optimizer:
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0005, amsgrad=True) #RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
    text_image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="{}/weights-improvement_image_text.hdf5".format(folder)
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
    filepath="{}/weights-improvement_image_text.hdf5".format(folder)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    callbacks_list = [reduce_lr, checkpoint]
    history = text_image_model.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)