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
from keras.layers import Dense, Reshape, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.optimizers import Adam
from utils import *

CLASSES = {0: 'bar', 1: 'gel', 2: 'map', 3: 'network', 4: 'plot',
         5: 'text', 6: 'box', 7: 'heatmap',8: 'medical', 9: 'nxmls', 10: 'screenshot',
         11: 'topology', 12: 'diagram', 13: 'histology', 14: 'microscopy',
         15: 'photo', 16: 'sequence', 17: 'tree', 18: 'fluorescence', 19: 'line',
         20: 'molecular', 21: 'pie', 22: 'table'}

class_names = ['bar', 'gel', 'network', 'plot', 'histology', 'sequence',
                           'line', 'molecular']
num_class = len(class_names)
NN_Class_Names = dict(enumerate(class_names))
#Set parameters
create_word_vec = False
seed = 0
shape = (106,106)#image shape
max_text_len = 200
vocabulary_size = 10000
csv_fname = 'image_list.csv'
filters = ''
use_trained_weights = False
use_glove = False

def Generate_Model(image = True, text = True):
    #set up our text input
    if image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=shape[0], 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        embedded_text = embedding(input_text)
        text_embedding = Reshape((max_text_len,shape[0],1))(embedded_text)
        #set up our image input
        image_input = Input(shape, name='image_input')
        image_reshaped = Reshape((shape[0],shape[1],1))(image_input)
        '''merge our data for early fusion. Essentially we are just concatenating our data together'''
        merged = concatenate([text_embedding, image_reshaped], axis=1, name='merged')
        x = get_CNN()(merged)
        output = Dense(8, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[input_text, image_input], outputs=[output])
    if image and not text:
        image_input = Input(shape, name='image_input')
        image_reshaped = Reshape((shape[0],shape[1],1))(image_input)
        x = get_CNN()(image_reshaped)
        output = Dense(8, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[input_text], outputs=[output])
    if not image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=106, 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        embedded_text = embedding(input_text)
        text_embedding = Reshape((max_text_len,shape[0],1))(embedded_text)
        x = get_CNN()(text_embedding)
        output = Dense(num_class, activation='softmax', name='output')(x)
        classificationModel = Model(inputs=[image_input], outputs=[output])
    return classificationModel
    '''Returns our predictions'''
def test(model, data, image = True, text = True):
    if image and text:
        X_text_test, X_image_test, y_test = data
        model.evaluate(x=[X_text_test, X_image_test], y=[y_test])
        y_hat=model.predict([X_text_test, X_image_test]).argmax(axis=-1)
    elif image and not text:
        X_image_test, y_test = data
        model.evaluate(x=[X_image_test], y=[y_test])
        y_hat=model.predict([X_image_test]).argmax(axis=-1)
    elif not image and text:
        X_image_test, y_test = data
        model.evaluate(x=[X_image_test], y=[y_test])
        y_hat=model.predict([X_image_test]).argmax(axis=-1)
    
    cm = confusion_matrix([np.argmax(t) for t in y_test], y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    report = metrics.classification_report([np.argmax(t) for t in y_test], y_hat, target_names=list(NN_Class_Names.values()))
    print(report)
    return y_hat

#%% Section 0: Get data
if __name__ == '__main__':
    image_classes = [list(CLASSES.keys())[list(CLASSES.values()).index(x)] for 
                     x in class_names]  
    #get images
    data = get_images(csv_fname=csv_fname, classes=image_classes, uniform=True)
    #split into train test
    train, test = train_test_split(data, test_size=0.2, random_state=seed)


#%%Section 1 create embeddings
    #Section 1 create embeddings - Optional
    if create_word_vec:
        from w2v_keras import w2v_keras
        w2v = w2v_keras(vocabulary_size=vocabulary_size, window_size=3, filters='')
        w2v.fit(data['caption'])
#%% Generate our sequences from text
    #set up tokenizer
    tokenizer = Tokenizer(num_words= vocabulary_size, lower=True, filters=filters)
    tokenizer.fit_on_texts(filter_sentences(data['caption']))
    #get our Training and testing set ready
    X_text_train = caption2sequence(filter_sentences(train['caption'].copy(), flatten=False), tokenizer, max_text_len)
    X_text_test = caption2sequence(filter_sentences(test['caption'].copy(), flatten=False), tokenizer, max_text_len)
    X_image_train = get_images_from_df_one_channel(train, shape)/255
    X_image_test = get_images_from_df_one_channel(test, shape)/255
    y_train = np.array(train['class_id']).reshape((-1, 1))
    y_test = np.array(test['class_id']).reshape((-1, 1))
    onehotencoder = OneHotEncoder()
    y_train = onehotencoder.fit_transform(y_train).toarray()
    y_test = onehotencoder.transform(y_test).toarray()
    if use_glove:
        embedding_weights = load_glove('/home/kevin/Downloads', take(vocabulary_size, tokenizer.word_index.items()))
    elif use_trained_weights:#get weights from w2v model directly if trained.
        embedding_weights = mapTokensToEmbedding(w2v.get_embeddings(), 
                                                 tokenizer.word_index)
    else:
        #load embedding weights from file
        location = 'w2v_embeddings.json'
        import json
        with open(location) as f:
            embs = json.load(f)
            emb = json.loads(embs)
        embedding_weights = mapTokensToEmbedding(emb, tokenizer.word_index)
#%% Train on images + text
    classificationModel = Generate_Model(image = True, text=True)
    #set optimizer:
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0005, amsgrad=True) #RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
    classificationModel.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="early_fusion_weights/weights-improvement_image_text.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    callbacks_list = [reduce_lr, checkpoint]
    history = classificationModel.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
    test(classificationModel, zip(X_text_train, X_image_train, y_train))
#%%Test with Image + Noise
    #Testing with noise:
    num_test_samples = len(test)
    text_noise = np.random.rand(num_test_samples,max_text_len)
    image_noise = np.array([np.random.rand(shape[0],shape[1]) for x in range(num_test_samples)])
    test(classificationModel, zip(text_noise, X_image_train, y_train))
#%% Test with Noise + Text
    test(classificationModel, zip(X_text_train, image_noise, y_train))
#%% Test with Noise + Noise
    test(classificationModel, zip(text_noise, image_noise, y_train))
#%% Train with only images
    '''
    train with only images:
    '''
    #set up our image input
    image_input = Input((106,106), name='image_input')
    image_reshaped = Reshape((106,106,1))(image_input)
    x = get_CNN()(image_reshaped)
    output = Dense(8, activation='softmax', name='output')(x)
    classificationModel = Model(inputs=[image_input], outputs=[output])
    classificationModel.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="early_fusion_weights/weights-improvement_image_only.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, reduce_lr]
    history = classificationModel.fit([X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
    
    classificationModel.evaluate(x=[X_image_test], y=[y_test])
    y_hat=classificationModel.predict([X_image_test]).argmax(axis=-1)
    cm = confusion_matrix([np.argmax(t) for t in y_test], y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    report = metrics.classification_report([np.argmax(t) for t in y_test], y_hat, target_names=list(NN_Class_Names.values()))
    print(report)
    '''
    train with only text:
    '''
    #set up our text input
    input_text = Input(shape=(200,), name='text_input')
    embedding = Embedding(input_length=200, input_dim=vocabulary_size, output_dim=106, 
                           name='embedding_layer', trainable=False, weights=[embedding_weights])
    embedded_text = embedding(input_text)
    text_embedding = Reshape((200,106,1))(embedded_text)
    x = get_CNN()(text_embedding)
    output = Dense(8, activation='softmax', name='output')(x)
    classificationModel = Model(inputs=[input_text], outputs=[output])
    classificationModel.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="early_fusion_weights/weights-improvement_text_only.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, reduce_lr]
    history = classificationModel.fit([X_text_train],[y_train], 
                             epochs=150, batch_size=32, validation_split=0.1, callbacks=callbacks_list)
    
    classificationModel.evaluate(x=[X_text_test], y=[y_test])
    y_hat=classificationModel.predict([X_text_test]).argmax(axis=-1)
    cm = confusion_matrix([np.argmax(t) for t in y_test], y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    report = metrics.classification_report([np.argmax(t) for t in y_test], y_hat, target_names=list(NN_Class_Names.values()))
    print(report)
    #lets extract the embeddings
    #extract_embeddings = Model(inputs=input_layer, outputs=embedded_text)
    #sequence_embeddings = extract_embeddings.predict(padded_seq[:1])
    