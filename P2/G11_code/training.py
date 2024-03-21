
from G11_code.data_collection import flatten
from sklearn.decomposition import PCA 
import xgboost as xgb


def split_by_cat(sentence_embeddings_by_cat: list, summary_sentence_indices_by_cat: list):
    category_group = zip(sentence_embeddings_by_cat, summary_sentence_indices_by_cat)
    
    train_emb_by_cat = list()
    train_ind_by_cat = list()
    test_emb_by_cat = list() 
    test_ind_by_cat = list() 

    for sentence_embeddings, sentence_indices in category_group: 
        train_emb, test_emb, train_ind, test_ind = split(sentence_embeddings, sentence_indices, 0.8)
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
    X = flatten(X)
    Y = flatten(Y)
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

'''
training(Dtrain,Rtrain,args)
    @input training document collection Dtrain, reference extracts Rtrain, and optional
    arguments on the classification process
    @behavior learns a classifier to predict the presence of a sentence in the summary
    @output classification model
'''
def training(Dtrain: list, Rtrain: list, **args):
    model_name =  ('model_name' in args and args['model_name']) or 'XGBoost'

    X_train, Y_train = get_XY(Dtrain, Rtrain)

    if args["use_pca"]: 
        pca = fit_PCA(X_train, n_components=args["n_components"])
        X_train = transform_PCA(pca, X_train)

    if model_name == "XGBoost": 
        model = xgb.XGBClassifier() 
        model.fit(X_train, Y_train)
    elif model_name == "LSTM": 
        model = None
    else:
        raise ValueError("Currently we only support the following models for summarization:\n→ XGBoost\n→ BM-25\n→ LSTM")
    
    return model