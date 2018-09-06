#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 14:29:53 2018

@author: kevin
"""
home_server = False
import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, dot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, skipgrams, make_sampling_table
from keras.layers.embeddings import Embedding
from keras.engine.input_layer import Input
import pandas as pd
import random, collections
from keras.initializers import RandomNormal, TruncatedNormal, Zeros, Constant
from keras.regularizers import l1, l2
from keras.layers.core import Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import itertools
import nltk
from nltk.corpus import stopwords
import keras

import os
if home_server:
    if os.getcwd() != '/home/kevin/Downloads/Aim_1':
        os.chdir('/home/kevin/Downloads/Aim_1')
else:
    if os.getcwd() != '/home/kevin/Documents/Lab/Aim_1':
        os.chdir('/home/kevin/Documents/Lab/Aim_1')


vocabulary_size = 10000
window_size = 5
epochs = 100
valid_size = 16
vector_dim = 106
valid_window = 100
filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890\n\r→▶≡≈†p‡'
embedding_file_location = 'embeddings_weights'
neg_samples = 10.0
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
sequence_size = 10000
#generate skipgrams
print('creating sents ({} rows)'.format(len(data['caption'])))
#sents = newsgroups_train.data
sents = [nltk.sent_tokenize(s) for s in data['caption']]
sents = list(itertools.chain.from_iterable(sents))
sents = [x.strip() for x in sents]

print('filtered sents')
filtered_sents = []
import re
stop_words_re = re.compile(r'\b(?:%s)\b' % '|'.join(set(stopwords.words('english'))))
for sent in sents:
    s = " ".join(re.findall("[a-zA-Z-]+", sent))
    s = re.sub(stop_words_re, '', s)
    s = re.sub(' +',' ',s)
    if len(s.split()) > 2:
        filtered_sents.append(s)

print('tokenizing sents ({} sentences)'.format(len(sents)))
tokenizer = Tokenizer(num_words= vocabulary_size, lower=True, filters=filters)
tokenizer.fit_on_texts(sents)
word_index_inv = {v: k for k, v in tokenizer.word_index.items()}
sequences = tokenizer.texts_to_sequences(sents)    

sampling_table = make_sampling_table(vocabulary_size, sampling_factor=0.001)
print('generating couples')
couples = []
labels = []
for seq in sequences:
    c,l = skipgrams(seq, vocabulary_size=vocabulary_size, 
            window_size=window_size, shuffle=True, sampling_table=sampling_table, 
            negative_samples=neg_samples)
    couples.extend(c)
    labels.extend(l)

word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print('building model')
#build model
stddev = 1.0 / vector_dim
initializer = TruncatedNormal(mean=0.0, stddev=stddev, seed=None)
reduce_lr = ReduceLROnPlateau(mode='auto', monitor='acc', factor=0.2, patience=2, min_lr=1e-5, verbose=1)
regularizer = l2(0.)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
input_target = Input((1,))
input_context = Input((1,))
embedding = Embedding(vocabulary_size, vector_dim, input_length=1, name='embedding', 
                      embeddings_initializer=initializer)

target = embedding(input_target)
context = embedding(input_context)
# setup a cosine similarity operation which will be output in a secondary model
similarity = dot([target, context], normalize=True, axes=-1)#we can see that we don't use the similarity for anything but checking
# now perform the dot product operation to get a similarity measure
dot_product = dot([target, context], normalize=False, axes=-1)
dot_product = Reshape((1,), input_shape=(1,1))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)
'''
input_target = Input((2,))
input_context = Input((2,))
dot_product = dot([input_target, input_context], normalize=True, axes=1)
test_model = Model(inputs=[input_target, input_context], outputs=dot_product)

input_target = Input((1,))
embedding = embedding(input_target)
test_model = Model(inputs=[input_target], outputs=[embedding])

'''
class SimilarityCallback:
    def run_sim(self, word, num_similar = 10):
        if word in tokenizer.word_index:
            valid_word = tokenizer.word_index[word]
            top_k = num_similar  # number of nearest neighbors
            sim = self._get_sim(valid_word)
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % word
            for k in range(top_k):
                close_word = word_index_inv[nearest[k]]
                log_str = '%s %s %f,' % (log_str, close_word, sim[nearest[k]])
            print(log_str)
        else:
            print('key not in vocab')
            
    def _get_sim(self, valid_word_idx):
        sim = validation_model.predict([[valid_word_idx]*(vocabulary_size), list(range(vocabulary_size))]).flatten()
        return sim
      
class get_similar(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for ex in valid_examples:
            sim_cb.run_sim(word_index_inv[ex], 5)

sim_cb = SimilarityCallback()
similar = get_similar()

df = pd.DataFrame({'word_target': word_target, 'word_context': word_context, 
                   'labels': labels})
#df = df.sample(n=200)
#df = df.sample(frac=1.0)
history = model.fit([df['word_target'],df['word_context']], df['labels'], 
                    batch_size=10000, epochs=200, callbacks=[reduce_lr, similar])




#plot
embedding_weights = embedding.get_weights()

print('performing TSNE')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(embedding_weights[0])

#pca = PCA(n_components=2)
#pca_result = pca.fit_transform(embedding_weights[0])

embeddings_df = pd.DataFrame({'words': [word_index_inv[x] for x in range(1,10000)], 'x1': tsne_results[1:,0], 'x2': tsne_results[1:,1]})
embeddings_df.to_csv('embeddings_df.csv')


import matplotlib.pyplot as plt
head = 600
plt.scatter(embeddings_df.head(head)['x1'], embeddings_df.head(head)['x2'])
for i, txt in enumerate(embeddings_df.head(head)['words']):
    plt.annotate(txt, (embeddings_df.iloc[i]['x1'],embeddings_df.iloc[i]['x2']))

embeddings_df = pd.read_csv('embeddings_df.csv')
embeddings_df = embeddings_df.sample(frac=1.0)
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
#ax.scatter(embeddings_df.loc[:100,'x1'], embeddings_df.loc[:100,'x2'])
#for i, txt in enumerate(embeddings_df['words']):
#    ax.annotate(txt, (embeddings_df.loc[i, 'x1'],embeddings_df.loc[i, 'x2']))
#
#fig.savefig('scatter_plot_w2v.png')
#plt.close(fig)    # close the figure
from w2v_keras import w2v_keras
w2v = w2v_keras()
w2v.fit(data['caption'])

#gensim implementation
import gensim
import nltk
sents = newsgroups_train.data
sents_tok = [nltk.word_tokenize(x) for x in sents]
# build vocabulary and train model

model = gensim.models.Word2Vec(
    size=106,
    window=3,
    min_count=1,
    workers=10,
    sg = 1,
    negative=10)
model.build_vocab(sents_tok)
model.train(sents_tok, total_examples=len(sents_tok), epochs=10, compute_loss=True)
model.most_similar('Canada')
model.vocab
model.wv.vocab
model.get_latest_training_loss()


