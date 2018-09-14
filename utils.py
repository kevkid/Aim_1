#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 17:37:38 2018

@author: kevin
Utils file
Place all common methods here
"""
import os
import get_images_from_db
import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Dropout, Flatten
from keras.preprocessing.sequence import pad_sequences
from itertools import islice
import itertools
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pycocotools.coco import COCO
import pylab

def get_data(csv_fname = 'image_list.csv',from_db = False, classes = list(range(0,23)), uniform = True):
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
def get_images_from_df(df, shape):
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
def get_images_from_df_one_channel(df, shape):
    images = []
    for i, (index, sample) in enumerate(df.iterrows()):
            #read image
            image = cv2.imread(sample["location"])
            image = cv2.resize(image, shape)
            if len(image.shape) == 3:#if there is no channel data
                image = image[:,:,1]
            images.append(image)
    return np.array(images)

def caption2sequence(captions, tokenizer, max_text_len = 200, padding = 'post'):
    for i, caption in enumerate(captions):
        sequence = tokenizer.texts_to_sequences([caption])    
        padded_seq = pad_sequences(sequence, maxlen=max_text_len, padding=padding)
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

def get_CNN_TEXT():
    CNN_Model = Sequential()
    #block 1
    CNN_Model.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv1'))
    CNN_Model.add(Conv1D(32, kernel_size=2, activation='relu', padding='valid', kernel_initializer='he_normal', name='block1_conv2'))
    CNN_Model.add(MaxPooling1D(pool_size=2, name='block1_pool'))
    CNN_Model.add(Dropout(0.2))
    #block 2
    CNN_Model.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv1'))
    CNN_Model.add(Conv1D(64, kernel_size=3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block2_conv2'))
    CNN_Model.add(MaxPooling1D(pool_size=3, name='block2_pool'))
    CNN_Model.add(Dropout(0.2))    
    #Flatten
    CNN_Model.add(Flatten())
    #Dense section
    CNN_Model.add(Dense(1024, activation='relu', name='dense_layer1'))
    CNN_Model.add(Dropout(0.5))
    CNN_Model.add(Dense(512, activation='relu', name='dense_layer2'))
    CNN_Model.add(Dropout(0.5))
    return CNN_Model

def load_glove(GLOVE_DIR, word_index, tokenizer, vocabulary_size = 10000, n_components = 106, EMBEDDING_DIM=300):
    from sklearn.decomposition import PCA
    #load embeddings
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
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embedding_matrix)
    return pca_result

def mapTokensToEmbedding(embedding, word_index, vocabulary_size):#function to map embeddings from my w2v
    embedding_matrix = np.zeros((vocabulary_size, len(embedding['-UNK-'])))
    for word, index in word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embedding.get(word)#Takes index of word 
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix
'''
    Flattening means to take the samples and extract the sentences regardless 
    of structure of the samples
'''
def filter_sentences(documents, flatten = True):
    print('filtering sentences')
    if flatten:
        sents = [nltk.sent_tokenize(s) for s in documents]
        sents = list(itertools.chain.from_iterable(sents))
    else:
        sents = documents
    sents = [x.strip() for x in sents]
    print('filtering sents and removing stopwords')
    filtered_sents = []
    import re
    stop_words = set([ "a", "about", "above", "after", "again", "against", 
                  "all", "am", "an", "and", "any", "are", "as", "at", "be", 
                  "because", "been", "before", "being", "below", "between", 
                  "both", "but", "by", "could", "did", "do", "does", "doing",
                  "down", "during", "each", "few", "for", "from", "further", 
                  "had", "has", "have", "having", "he", "he'd", "he'll", "he's", 
                  "her", "here", "here's", "hers", "herself", "him", "himself", 
                  "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", 
                  "in", "into", "is", "it", "it's", "its", "itself", "let's",
                  "me", "more", "most", "my", "myself", "nor", "of", "on", 
                  "once", "only", "or", "other", "ought", "our", "ours", 
                  "ourselves", "out", "over", "own", "same", "she", "she'd", 
                  "she'll", "she's", "should", "so", "some", "such", "than", 
                  "that", "that's", "the", "their", "theirs", "them",
                  "themselves", "then", "there", "there's", "these", "they", 
                  "they'd", "they'll", "they're", "they've", "this", "those", 
                  "through", "to", "too", "under", "until", "up", "very", 
                  "was", "we", "we'd", "we'll", "we're", "we've", "were", 
                  "what", "what's", "when", "when's", "where", "where's", 
                  "which", "while", "who", "who's", "whom", "why", "why's", 
                  "with", "would", "you", "you'd", "you'll", "you're", 
                  "you've", "your", "yours", "yourself", "yourselves", "ing" ] + 
                    stopwords.words('english'))
    stop_words_re = re.compile(r'\b(?:%s)\b' % '|'.join(stop_words))
    for sent in sents:
        s = sent.lower()
        s = " ".join(re.findall("[a-zA-Z-]+", s))
        s = re.sub(stop_words_re, '', s)
        s = re.sub(' +',' ',s)
        s = re.sub('\'s','',s)
        #s = " ".join(nltk.PorterStemmer().stem(x) for x in s.split())#may not need to stem...
#            if len(s.split()) > 2:
        filtered_sents.append(s)
    return filtered_sents

def load_PMC(csv_fname, classes, uniform = True):
    return get_data(csv_fname=csv_fname, classes=classes, uniform=uniform)

def load_COCO(location, class_names = ['cat','dog']):
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)
    #####################Get Training data
    dataDir=location#'/media/kevin/1142-5B72/coco_dataset'
    #dataDir='/home/kevin/Documents/Lab/coco_dataset'
    dataType='val2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    #I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # initialize COCO api for caption annotations
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)
    #encapsulate this function
    def build_df_coco(imgids, class_ID):
        ImgIdsDict = {}
        for imgid in imgids:
            annIds = coco_caps.getAnnIds(imgIds=imgid);
            captions = [ann['caption'] for ann in coco_caps.loadAnns(annIds)]
            img = coco.loadImgs(imgid)[0]
            ImgIdsDict[imgid] = {
                    'class_id':class_ID,
                    'caption':' '.join(captions),
                    'location': '%s/images/%s/%s'%(dataDir,dataType,img['file_name'])
                    }
        df = pd.DataFrame.from_dict(ImgIdsDict, orient='index')
        return df
    
    #get images of dogs:
    catIds = coco.getCatIds(catNms=class_names);#intersection of images with all categories
    img_ids = [coco.getImgIds(catIds=x) for x in catIds]#gets our image ids
    filtered_ids_dfs = []
    for i, img_id in enumerate(img_ids):
        tmp = img_ids.copy()
        tmp.pop(i)
        tmp = [item for sublist in tmp for item in sublist]
        filtered = [img for img in img_id if img not in tmp]
        filtered_ids_dfs.append(build_df_coco(filtered, i))
    
    data_df = pd.concat(filtered_ids_dfs)
    return data_df
    
def get_model_data(data, seed = 0, test_size = 0.2, vocabulary_size = 10000, filters = '', max_text_len=200, shape = (106,106)):
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
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
    return (X_text_train, X_text_test, X_image_train, X_image_test, y_train, y_test, tokenizer)