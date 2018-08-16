#Scrap code:
def maxSentLen(tokens):
    maxlen = 0
    for s in tokens:
        l = len(s)
        if l > maxlen:
            maxlen = l
    return maxlen

#make our training data:
def skip_gram(tokens, grams = 1):#can reuse tokens from first tokenization
    '''
    Our data is as follows:
        X is the center word (in one hot form)
        y is a context word (in one hot form)
        so we will have many context words for each center word
        we have to train our network on this X and y
        Our neural network will learn a representation of the words similarities
        in that hidden layer.
    '''
    X = []
    y = []
    grams = 2
    list_grams = list(range(-grams, grams+1))
    list_grams.remove(0)
    for tok_sent in tokens:#get token sentence
        for i, token in enumerate(tok_sent):#for each token in each sentence
            for gram in list_grams:
                if i+gram > -1 and i+gram < len(tok_sent):
                    X.append(tokens_1hot[token])
                    y.append(tokens_1hot[tok_sent[i+gram]])
    return X, y
'''
    Tokenizer
    num_words: the maximum number of words to keep, based on word frequency.
    Only the most common num_words words will be kept.
    lower: boolean. Whether to convert the texts to lowercase.
    '''
    import nltk
    from nltk.corpus import stopwords
    nltk.download('punkt')
    #captions sentence tokenized
    captions = []
    tokens = []
    for doc in data.loc[:200,'caption']:
        #Sentence tokenize the document
        sentence_tokenized_caption = nltk.sent_tokenize(doc)
        #Tokenize sentences & normalize sentences
        word_tokenized_caption = [nltk.word_tokenize(word.lower()) for word in sentence_tokenized_caption]
        captions.append(word_tokenized_caption)
        #create our tokens list to make our vocabulary
        tokens.extend(word_tokenized_caption)
    max_len = maxSentLen(tokens)
    #create our vocabulary
    import itertools
    tokens_all = list(itertools.chain.from_iterable(tokens))
    #get stop words
    nltk.download('stopwords')
    stop_words = list(set(stopwords.words('english')))
    stop_words.extend([x for x in '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890-\'\n\r'])
    tokens_all_no_stop_words = [w for w in tokens_all if not w in stop_words]
    
    from nltk import FreqDist
    fdist = FreqDist(tokens_all_no_stop_words)
    vocabulary = [x[0] for x in fdist.most_common(vocabulary_size)]
    
    #create our 1hot arrays so we can start to encode everything into sequences
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    tokens_enc = le.fit_transform(vocabulary)
    ohe = OneHotEncoder(n_values=vocabulary_size)
    tokens_1hot = ohe.fit_transform(np.array(tokens_enc).reshape(-1,1)).toarray()
    import copy
    captions_sequences = copy.deepcopy(captions)    
    sentences = []
    
    for i, caption in enumerate(captions_sequences):
        for j, sentence in enumerate(caption):
            seq = []
            for token in sentence:
                tok = np.where(le.classes_ == token)[0]            
                if len(tok) < 1:#should we be setting words that arent in a sequence to -1 or just removing them all together?
                    tok = -1
                else:
                    tok = tok[0]                
                seq.append(tok)
            #seq.extend((max_len-len(seq))*[-1])#Padding
            captions_sequences[i][j] = seq
            sentences.append(seq)
            
    X_text, y_text = skip_gram(sentences, grams=1)
    
    model = Sequential()
    model.add(Dense(106, input_dim=vocabulary_size))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_text, y_text, epochs=1000, validation_split=0.1)
    
    score = model.evaluate(X_test_text, y_test_text_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    weights = model.layers[0].get_weights()[0]
    #bias = model.layers[0].get_weights()[1]
    
    #graph the embeddings
    plt.scatter(weights[:1000,0], weights[:1000,1])
    for i in range(1,1001):
        plt.annotate(word_index_inv[i], (weights[i-1,0],weights[i-1,1]))
    
    
    
    
    
    #ONE_HOT = one_hot(text=data['caption'], n=vocabulary_size, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890-')
    
    tokenizer = Tokenizer(num_words= vocabulary_size, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~1234567890-')
    
    tokenizer.fit_on_texts(data['caption'])
    word_index_inv = {v: k for k, v in tokenizer.word_index.items()}
    sequences = tokenizer.texts_to_sequences(data['caption'])    
    padded_seq = pad_sequences(sequences, maxlen=200, padding='post')
    couples, labels = skipgrams(padded_seq[0], vocabulary_size=vocabulary_size, categorical=True, window_size=1)
    
    X_train_text = pad_sequences(sequences, maxlen=200)
    #text_labels = to_categorical(data['class_id'])
    ohe = OneHotEncoder()
    y_train_text_labels = ohe.fit_transform(np.array(train['class_id']).reshape(-1,1)).toarray()
    
    sequences = tokenizer.texts_to_sequences(test['caption'])
    X_test_text = pad_sequences(sequences, maxlen=200)
    #text_labels = to_categorical(data['class_id'])
    y_test_text_labels = ohe.transform(np.array(test['class_id']).reshape(-1,1)).toarray()
    
    model = Sequential()
    model.add(Embedding(vocabulary_size, 2, input_length=200))
    model.add(Flatten())
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_text, y_train_text_labels, epochs=1000, validation_split=0.1)
    
    score = model.evaluate(X_test_text, y_test_text_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    weights = model.layers[0].get_weights()[0]
    #bias = model.layers[0].get_weights()[1]
    
    #graph the embeddings
    plt.scatter(weights[:1000,0], weights[:1000,1])
    for i in range(1,1001):
        plt.annotate(word_index_inv[i], (weights[i-1,0],weights[i-1,1]))
    

    

    X_train = get_images_from_df_one_channel(train)/255
    X_test = get_images_from_df_one_channel(test)/255
    y_train = np.array([class_id for class_id in train.loc[: ,'class_id']]).reshape((-1, 1))
    y_test = np.array([class_id for class_id in test.loc[: ,'class_id']]).reshape((-1, 1))
    
    
    onehotencoder = OneHotEncoder()
    y_train = onehotencoder.fit_transform(y_train).toarray()
    y_test = onehotencoder.transform(y_test).toarray()
    '''Conv Net'''
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
    y_test = onehotencoder.transform(y_test).toarray()#reset back to normal for prediction
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


