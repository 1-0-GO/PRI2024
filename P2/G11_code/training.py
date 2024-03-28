
from G11_code.data_collection import flatten
from G11_code.NNModel import NNModel
from G11_code.LSTMModel import LSTMModel
from sklearn.decomposition import PCA 
from sklearn.utils import class_weight
import xgboost as xgb
import numpy as np
import pandas as pd
import keras 
import umap
from itertools import groupby


def split_by_cat(sentence_embeddings_by_cat: list, summary_sentence_indices_by_cat: list):
    category_group = zip(sentence_embeddings_by_cat, summary_sentence_indices_by_cat)
    
    train_emb_by_cat = list()
    train_ind_by_cat = list()
    test_emb_by_cat = list() 
    test_ind_by_cat = list() 

    train_doc_indices = list()
    test_doc_indices = list()

    for sentence_embeddings, sentence_indices in category_group: 
        train_emb, test_emb, train_ind, test_ind = split(sentence_embeddings, sentence_indices, 0.8)
        # indices 
        t1, t2 = split(range(len()))
        train_emb_by_cat.append(train_emb)
        train_ind_by_cat.append(train_ind)
        test_emb_by_cat.append(test_emb)
        test_ind_by_cat.append(test_ind)

    return train_emb_by_cat, test_emb_by_cat, train_ind_by_cat, test_ind_by_cat 


def get_XY(sentence_embeddings_by_cat: list, summary_sentence_indices_by_cat: list): 
    X = flatten(sentence_embeddings_by_cat)
    Y = list()
    for i, indices in enumerate(flatten(summary_sentence_indices_by_cat)):
        y = [0] * len(X[i])
        for k in indices: 
            y[k] = 1
        Y.append(y)
    #X = flatten(X)
    #Y = flatten(Y)
    return X, Y

def split(X: list, Y: list, train_ratio: float):
    n = int(len(X) * train_ratio)
    X_train = X[:n]
    X_test = X[n:]
    Y_train = Y[:n]
    Y_test = Y[n:]
    return X_train, X_test, Y_train, Y_test

def fit_PCA(X, n_components): 
    pca = PCA(n_components=n_components)
    return pca.fit(X)

def transform_PCA(pca, X):
    return pca.transform(X)

def fit_UMAP(X, n_components): 
    reducer = umap.UMAP(n_components=n_components)
    return reducer.fit(X)

def transform_UMAP(reducer:umap.UMAP, X): 
    return reducer.transform(np.array(X))

'''
training(Dtrain,Rtrain,args)
    @input training document collection Dtrain, reference extracts Rtrain, and optional
    arguments on the classification process
    @behavior learns a classifier to predict the presence of a sentence in the summary
    @output classification model
'''
def training(Dtrain: list, Rtrain: list, **args):
    model_name =  ('model_name' in args and args['model_name']) or 'XGBoost'
    use_extracted_features = ('use_extracted_features' in args and args['use_extracted_features']) or False
    use_val = ('use_val' in args and args['use_val']) or False
    
    if use_val:
        X_val = ('X_val' in args and args['X_val'])
        y_val = ('y_val' in args and args['y_val'])
        
    if use_extracted_features:
        X_train, Y_train = Dtrain, Rtrain
    else:
        X_train, Y_train = get_XY(Dtrain, Rtrain)

        
    n_components = ('n_components' in args and args['n_components']) or (X_train.shape[1] if (type(X_train)==pd.core.frame.DataFrame) else len(X_train[0]))

    if (model_name != "LSTM") and (not use_extracted_features): 
        X_train = flatten(X_train)
        Y_train = flatten(Y_train)
        X_train = flatten(X_train)
        Y_train = flatten(Y_train)

    if "use_pca" in args: 
        if model_name != "LSTM": 
            pca = fit_PCA(X_train, n_components=args["n_components"])
            X_train = transform_PCA(pca, X_train)
        else: 
            pca = fit_PCA(flatten(X_train), n_components=args["n_components"])
            X_train= [transform_PCA(pca, x) for x in X_train]
    elif "use_umap" in args: 
        reducer = args["reducer"]
        X_train = transform_UMAP(reducer, X_train)

    if model_name == "XGBoost": 
        model = xgb.XGBClassifier() 
        model.fit(X_train, Y_train)
    elif model_name == "NN":
        if 'n_components' in args: 
            feature_length = args['n_components']
        else: 
            feature_length = len(X_train[0])
        model = NNModel(feature_length)
        model.compile(loss='binary_crossentropy', 
                 optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
                 metrics=['AUC'])
        if use_val:
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
            class_weights = dict(enumerate(class_weights))
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            model.fit(np.array(X_train), np.array(Y_train), epochs=1500,  class_weight=class_weights, validation_data=(np.array(X_val), np.array(y_val)), callbacks=[early_stopping], verbose=True)
        else:
            model.build(input_shape=(None, feature_length))
            model.summary()
            model.fit(np.array(X_train), np.array(Y_train), epochs=100, verbose=2)
    elif model_name == "LSTM": 
        if 'n_components' in args: 
            feature_length = args['n_components']
        else: 
            feature_length = len(X_train[0])
        model = LSTMModel(128, feature_length)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["AUC"])
        #model.build(input_shape=(None, None, feature_length))
        #print(model.summary())
        Y_train = [y for _, y in sorted(zip(X_train, Y_train), key=lambda x: len(x[0]))]
        X_train.sort(key=len)
        X_train_groups = [list(g) for k, g in groupby(X_train, key=len)]
        Y_train_groups = [list(g) for k, g in groupby(Y_train, key=len)]
        epochs = 100
        for i in range(epochs):
            for X, y in zip(X_train_groups, Y_train_groups):
                X = np.array(X)
                X = X.reshape(len(X), len(X[0]), feature_length)
                y = np.array(y)
                y = y.reshape(len(y), len(y[0]), 1)
                history = model.fit(X, y, epochs=1, batch_size=32, verbose=0, shuffle=True)
            avg_loss = np.mean(history.history['loss'][-100:])
            avg_auc = np.mean(history.history['auc'][-100:])
            print(f"Loss:{avg_loss} | AUC:{avg_auc}")
    else:
        raise ValueError("Currently we only support the following models for summarization:\n→ XGBoost\n→ BM-25\n→ LSTM")
    
    return model