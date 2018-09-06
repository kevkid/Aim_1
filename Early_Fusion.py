#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 23:12:23 2018

@author: kevin
"""

import os
if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
    os.chdir('/home/kevin/Documents/Lab/Aim_1')
import get_images_from_db
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from sklearn.metrics import confusion_matrix    
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, concatenate
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
from keras.optimizers import Adam, RMSprop
from itertools import islice
import itertools
import nltk
from nltk.corpus import stopwords
import random
import pandas as pd
CLASSES = {0: 'bar', 1: 'gel', 2: 'map', 3: 'network', 4: 'plot',
         5: 'text', 6: 'box', 7: 'heatmap',8: 'medical', 9: 'nxmls', 10: 'screenshot',
         11: 'topology', 12: 'diagram', 13: 'histology', 14: 'microscopy',
         15: 'photo', 16: 'sequence', 17: 'tree', 18: 'fluorescence', 19: 'line',
         20: 'molecular', 21: 'pie', 22: 'table'}

class_names = ['bar', 'gel', 'network', 'plot', 'histology', 'sequence',
                           'line', 'molecular']
NN_Class_Names = dict(enumerate(class_names))

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

#TODO: Make the images load using relative location
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
#TODO: Make the images load using relative location
def get_images_from_df_one_channel(df):
    images = []
    for i, (index, sample) in enumerate(df.iterrows()):
            #read image
            image = cv2.imread(sample["location"])
            image = cv2.resize(image, shape)
            if len(image.shape) == 3:#if there is no channel data
                image = image[:,:,1]
            images.append(image)
    return np.array(images)

def caption2sequence(captions, tokenizer):
    for i, caption in enumerate(captions):
        sequence = tokenizer.texts_to_sequences([caption])    
        padded_seq = pad_sequences(sequence, maxlen=200, padding='post')
        captions[i] = padded_seq.flatten()
    return captions

def take(n, iterable):#vocabulary size MUST be smaller than vocab generated from text
    "Return first n items of the iterable as a list"
    return dict(islice(iterable, n))
def get_CNN():
    CNN_Model = Sequential()
    #block 1
    CNN_Model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
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
    return CNN_Model
def load_glove(GLOVE_DIR, word_index):
    from sklearn.decomposition import PCA
    #load embeddings
    EMBEDDING_DIM = 300
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    print(np.shape(embedding_matrix))
    print('performing PCA')
    pca = PCA(n_components=106)
    pca_result = pca.fit_transform(embedding_matrix)
    return pca_result

def mapTokensToEmbedding(embedding, word_index):#function to map embeddings from my w2v
    embedding_matrix = np.zeros((vocabulary_size, len(embedding['-UNK-'])))
    for word, index in word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embedding.get(word)#Takes index of word 
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix
if __name__ == '__main__':
    #Section 0: Get data
    seed = 0
    shape = (106,106)#image shape
    image_classes = [list(CLASSES.keys())[list(CLASSES.values()).index(x)] for 
                     x in class_names]  
    #get images
    data = get_images(csv_fname='image_list.csv', classes=image_classes, uniform=True)
    #split into train test
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
#==============================================================================
    #Section 1 create embeddings
    vocabulary_size = 10000
    from w2v_keras import w2v_keras
    w2v = w2v_keras(vocabulary_size=vocabulary_size, window_size=3, filters='')
    w2v.fit(data['caption'], epochs=30)
    #w2v_model.fit(data.loc[:10000,'caption'])
    #embeddings = w2v_model.get_embeddings()
    '''
    fusion model should look like 2 inputs being text and image
    '''
    #generate our sequences from text
    #set up tokenizer
    tokenizer = Tokenizer(num_words= vocabulary_size, lower=True)
    
    tokenizer.fit_on_texts(w2v.filter_sentences(data['caption']))
    #get our Training and testing set ready
    X_text_train = caption2sequence(w2v.filter_sentences(train['caption'].copy(), flatten=False), tokenizer)
    X_text_test = caption2sequence(w2v.filter_sentences(test['caption'].copy(), flatten=False), tokenizer)
    X_image_train = get_images_from_df_one_channel(train)/255
    X_image_test = get_images_from_df_one_channel(test)/255
    y_train = np.array(train['class_id']).reshape((-1, 1))
    y_test = np.array(test['class_id']).reshape((-1, 1))
    onehotencoder = OneHotEncoder()
    y_train = onehotencoder.fit_transform(y_train).toarray()
    y_test = onehotencoder.transform(y_test).toarray()
    
    use_trained_weights = True
    use_glove = False
    if use_glove:
        embedding_weights = load_glove('/home/kevin/Downloads', take(vocabulary_size, tokenizer.word_index.items()))
    elif use_trained_weights:
        embedding_weights = mapTokensToEmbedding(w2v.get_embeddings(), 
                                                 tokenizer.word_index)
    else:
        #load embedding weights
        embedding_weights = np.genfromtxt('embedding_weights.csv', delimiter=',')
    #our model    
    #set optimizer:
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)#Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0005, amsgrad=False)
    #set up our text input
    input_text = Input(shape=(200,), name='text_input')
    embedding = Embedding(input_length=200, input_dim=vocabulary_size, output_dim=106, 
                           name='embedding_layer', trainable=False, weights=[embedding_weights])
    embedded_text = embedding(input_text)
    text_embedding = Reshape((200,106,1))(embedded_text)
    #set up our image input
    image_input = Input((106,106), name='image_input')
    image_reshaped = Reshape((106,106,1))(image_input)
    
    '''
    merge our data for early fusion. Essentially we are just concatenating our data together
    
    '''
    merged = concatenate([text_embedding, image_reshaped], axis=1, name='merged')
    x = get_CNN()(merged)
    output = Dense(8, activation='softmax', name='output')(x)
    
    classificationModel = Model(inputs=[input_text, image_input], outputs=[output])
    classificationModel.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # checkpoint
    filepath="early_fusion_weights/weights-improvement_image_text.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
    callbacks_list = [reduce_lr, checkpoint]
    history = classificationModel.fit([X_text_train, X_image_train],[y_train], 
                             epochs=150, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
    
    classificationModel.evaluate(x=[X_text_test, X_image_test], y=[y_test])
    y_hat=classificationModel.predict([X_text_test, X_image_test]).argmax(axis=-1)
    cm = confusion_matrix([np.argmax(t) for t in y_test], y_hat)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    report = metrics.classification_report([np.argmax(t) for t in y_test], y_hat, target_names=list(NN_Class_Names.values()))
    print(report)
    #Testing with noise:
    num_test_samples = 629
    text_noise = np.random.rand(629,200)
    image_noise = [np.random.rand(106,106) for x in range(629)]
    
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
    