#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Late Fusion
Created on Tue Sep 11 14:20:57 2018

@author: kevin
"""

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
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.optimizers import Adam
from utils import *

from keras import backend as K
#Our lambda functions
def mode(inputs):
    s = K.sum(inputs, axis=0)
    s = K.one_hot(K.argmax(s), K.shape(s)[0])
    return s

def averaging(inputs):
    s = K.sum(inputs, axis=0)
    s = K.softmax(s)
    return s
        

img_in = np.array([[0,1], [1,0]], dtype='int')
txt_in = np.array([[0,1], [1,0]], dtype='int')
other_in = np.array([[1,0], [0,1]], dtype='int')
img_input = Input((2,), name='image_input')
txt_input = Input((2,), name='text_input')
other_input = Input((2,), name='other_input')
x = Lambda(averaging)([img_input, txt_input, other_input])


model = Model(inputs=[img_input, txt_input, other_input], outputs=[x])

model.compile('adam', 'mse')
y_hat = model.predict(x=[img_in, txt_in, other_in])


#%% Load dependencies
import os
if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
    os.chdir('/home/kevin/Documents/Lab/Aim_1')
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import backend as K
from sklearn.metrics import confusion_matrix    
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Dense, Reshape, concatenate, Lambda, Average, Maximum, Add, Multiply
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
w2v_epochs = 1
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
        #set up our image branch
        image_input = Input(shape, name='image_input')
        image_reshaped = Reshape((shape[0],shape[1],1))(image_input)
        image_branch = get_CNN()(image_reshaped)
        image_branch_out = Dense(num_class, activation='softmax', name='image_branch_output')(image_branch)
        image_branch_model = Model(inputs=[image_input], outputs=[image_branch_out])
        #set up text branch
        text_branch = get_CNN()(text_embedding)
        text_branch_out = Dense(num_class, activation='softmax', name='text_branch_output')(text_branch)
        text_branch_model = Model(inputs=[input_text], outputs=[text_branch_out])
        outputs = [model.outputs[0] for model in [text_branch_model, image_branch_model]]
        #set up late fusion
        output = Average()(outputs)
        #output = Lambda(mode)([text_branch_out, image_branch_out])
        #output = Dense(num_class, activation='softmax', name='output')(output)
        classificationModel = Model(inputs=[input_text, image_input], outputs=[output])
    if image and not text:
        image_input = Input(shape, name='image_input')
        image_reshaped = Reshape((shape[0],shape[1],1))(image_input)
        x = get_CNN()(image_reshaped)
        output = Dense(num_class, activation='softmax', name='image_output')(x)
        classificationModel = Model(inputs=[image_input], outputs=[output])
    if not image and text:
        input_text = Input(shape=(max_text_len,), name='text_input')
        embedding = Embedding(input_length=max_text_len, input_dim=vocabulary_size, output_dim=106, 
                               name='embedding_layer', trainable=False, weights=[embedding_weights])
        embedded_text = embedding(input_text)
        text_embedding = Reshape((max_text_len,shape[0],1))(embedded_text)
        x = get_CNN()(text_embedding)
        output = Dense(num_class, activation='softmax', name='text_output')(x)
        classificationModel = Model(inputs=[input_text], outputs=[output])
    return classificationModel
    '''Returns our predictions'''
def test_model(model, data, image = True, text = True, class_names = None):
    if image and text:
        X_text_test, X_image_test, y_test = zip(*data)
        X_image_test = list(X_image_test)
        X_text_test = list(X_text_test)
        y_test = list(y_test)
        print(model.evaluate(x=[X_text_test, X_image_test], y=[y_test]))
        y_hat=model.predict([X_text_test, X_image_test]).argmax(axis=-1)
    elif image and not text:
        X_image_test, y_test = zip(*data)
        X_image_test = list(X_image_test)
        y_test = list(y_test)
        print(model.evaluate(x=[X_image_test], y=[y_test]))
        y_hat=model.predict([X_image_test]).argmax(axis=-1)
    elif not image and text:
        X_text_test, y_test = zip(*data)
        X_text_test = list(X_text_test)
        y_test = list(y_test)
        print(model.evaluate(x=[X_text_test], y=[y_test]))
        y_hat=model.predict([X_text_test]).argmax(axis=-1)
    
    cm = confusion_matrix([np.argmax(t) for t in y_test], y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    report = metrics.classification_report([np.argmax(t) for t in y_test], y_hat, target_names=class_names)
    print(report)
    return y_hat
#Our lambda functions
def mode(inputs):
    s = K.sum(inputs, axis=0)
    s = K.one_hot(K.argmax(s), num_class)
    s = K.softmax(s)
    return s

def averaging(inputs):
    s = K.sum(inputs, axis=0)
    s = K.softmax(s)
    return s

#%% Section 0: Get data
if __name__ == '__main__':
    print('Getting Data')
    image_classes = [list(CLASSES.keys())[list(CLASSES.values()).index(x)] for 
                     x in class_names]  
    #get images
    data = get_images(csv_fname=csv_fname, classes=image_classes, uniform=True)
    #split into train test
    train, test = train_test_split(data, test_size=0.2, random_state=seed)


#%%Section 0.5 create embeddings - Optional
    print('Section 0.5: create embeddings')
    if create_word_vec:
        from w2v_keras import w2v_keras
        w2v = w2v_keras(vocabulary_size=vocabulary_size, window_size=3, filters='')
        w2v.fit(data['caption'], epochs=w2v_epochs)
#%% Section 1: Generate our sequences from text
    print('Generate our sequences from text')
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
        embedding_weights = mapTokensToEmbedding(emb, tokenizer.word_index, vocabulary_size)
#%% Train on images + text
    print('Train on images + text')
    text_image_model = Generate_Model(image = True, text=True)
    #set optimizer:
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0005, amsgrad=True) #RMSprop(lr=0.001, rho=0.9, epsilon=1e-07, decay=0.0)
    text_image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="late_fusion_weights/weights-improvement_image_text.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    callbacks_list = [reduce_lr, checkpoint]
    history = text_image_model.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
#%%test on Image + text
    print('Test on Image + text')
    test_model(text_image_model, zip(X_text_test, X_image_test, y_test))
#%%Test with Image + Noise
    print('Testing Image + Noise')
    num_test_samples = len(test)
    text_noise = np.random.rand(num_test_samples,max_text_len)
    image_noise = np.array([np.random.rand(shape[0],shape[1]) for x in range(num_test_samples)])
    test_model(text_image_model, zip(text_noise, X_image_test, y_test), image = True, text = True)
#%% Test with Noise + Text
    print('Testing Noise + Text')
    test_model(text_image_model, zip(X_text_test, image_noise, y_test), image = True, text = True)
#%% Test with Noise + Noise
    print('Testing Noise + Noise')
    test_model(text_image_model, zip(text_noise, image_noise, y_test), image = True, text = True)
#%% Train with only images
    print('Train with only images')
    image_model = Generate_Model(image=True, text=False)
    image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="late_fusion_weights/weights-improvement_image_only.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, reduce_lr]
    history = image_model.fit([X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
#%%Test with only images
    print('Test with only images')
    test_model(image_model, zip(X_image_test, y_test), image = True, text = False)
    
#%%train with only text:    
    print('Train with only text')
    text_model = Generate_Model(image=False, text=True)
    text_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    filepath="late_fusion_weights/weights-improvement_text_only.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, reduce_lr]
    history = text_model.fit([X_text_train],[y_train], 
                             epochs=150, batch_size=32, validation_split=0.1, callbacks=callbacks_list)
#%%Test with only text
    print('Test with only text')
    test_model(text_model, zip(X_text_test, y_test), image = False, text = True)
    
#%%Merge text and image models into one big model
    outputs = [model.outputs[0] for model in [text_model, image_model]]
    output = Average()(outputs)    
    text_image_model = Model(inputs=[text_model.input, image_model.input], outputs=[output])
    text_image_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    test_model(text_image_model, zip(X_text_test, X_image_test, y_test), image = True, text = True)
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