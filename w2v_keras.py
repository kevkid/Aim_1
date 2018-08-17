#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec Keras implementation
Created on Mon Aug 13 11:33:41 2018

@author: kevin
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, skipgrams, make_sampling_table
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input

class w2v:
    
    def __init__(self, vocabulary_size = 10000, window_size = 2, 
                 epochs = 1000000, valid_size = 16, vector_dim = 106, valid_window = 100, 
                 filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890-\n\r'):
        self.vocabulary_size = vocabulary_size
        self.window_size = window_size
        self.vector_dim = vector_dim
        self.epochs = epochs
        self.valid_size = valid_size# Random set of words to evaluate similarity on.
        self.valid_window = valid_window # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        self.filters = filters
        
    def generate_skipgrams(self):
        sents = [sent for sent in self.documents]
        tokenizer = Tokenizer(num_words= self.vocabulary_size, lower=True, filters=self.filters)
        tokenizer.fit_on_texts(sents)
        self.word_index_inv = {v: k for k, v in tokenizer.word_index.items()}
        sequences = tokenizer.texts_to_sequences(sents)    
        padded_seq = pad_sequences(sequences, maxlen=200, padding='post')
        sampling_table = make_sampling_table(self.vocabulary_size)
        couples, labels = skipgrams(padded_seq.flatten(), self.vocabulary_size, 
                                    window_size=self.window_size, sampling_table=sampling_table, shuffle=False)
        
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        return word_target, word_context, labels
    
    def get_model(self, load_weights = False):
        input_target = Input((1,))
        input_context = Input((1,))
        embedding = Embedding(self.vocabulary_size, self.vector_dim, input_length=1, name='embedding')
        
        target = embedding(input_target)
        target = Reshape((self.vector_dim, 1))(target)#This target in parenthesis is telling the model to connect the reshape to target
        context = embedding(input_context)
        context = Reshape((self.vector_dim, 1))(context)
        # setup a cosine similarity operation which will be output in a secondary model
        similarity = dot([target, context], normalize=True, axes=0)#we can see that we don't use the similarity for anything but checking
        # now perform the dot product operation to get a similarity measure
        dot_product = dot([target, context], normalize=False, axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        # create the primary training model
        model = Model(inputs=[input_target, input_context], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        if load_weights:
            model.load_weights('embeddings_weights.h5')
        # create a secondary validation model to run our similarity checks during training
        validation_model = Model(inputs=[input_target, input_context], outputs=similarity)
        
        return model, validation_model
    
    def fit(self, documents):
        self.documents = documents
        word_target, word_context, labels = self.generate_skipgrams()
        
        self.model, self.validation_model = self.get_model()
        arr_1 = np.zeros((1,))
        arr_2 = np.zeros((1,))
        arr_3 = np.zeros((1,))
        for cnt in range(self.epochs):
            idx = np.random.randint(0, len(labels)-1)
            arr_1[0,] = word_target[idx]
            arr_2[0,] = word_context[idx]
            arr_3[0,] = labels[idx]
            loss = self.model.train_on_batch([arr_1, arr_2], arr_3)
            if cnt % 100 == 0:
                print("Iteration {}, loss={}".format(cnt, loss))
            if cnt % 100000 == 0:
                self.run_sim()
        np.savetxt("embedding_weights.csv", self.model.layers[2].get_weights()[0], delimiter=",")
        self.model.save_weights('embeddings_weights.h5')
        return self.model
    
    def run_sim(self):
        for i in range(self.valid_size):
            valid_word = self.word_index_inv[self.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(self.valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.word_index_inv[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    
    def _get_sim(self, valid_word_idx):
        sim = np.zeros((self.vocabulary_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(self.vocabulary_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
    def get_embeddings(self):
        return self.model.layers[2].get_weights()

