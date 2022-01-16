# LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
import re
import os.path


# FUNCTIONS

def data_preparation(df_ubication, label_column_name, test_size_rfc = 0.2, random_state_rfc = 42, stratify_rfc=False):

    '''
    This function loads the CSV file with the information an transforms it into a pandas dataframe.
    Then, divides the data into variables and labes (X and y) and makes the split into train and test.
    After that, the data will be transformed into numpy type to use them and will be returned.
    '''

    df = pd.read_csv(df_ubication, index_col=[0])
    X = df.drop(label_column_name, axis=1)
    y = df[label_column_name]
    if stratify_rfc == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_rfc, random_state=random_state_rfc, stratify=y)
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_rfc, random_state=random_state_rfc)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test

#---------------------------------------------------------------------------------------------------------------

def one_hot_encoder(ohe_ubication, X_train, X_test, sparse_ohe = False):

    '''
    This function creates the OneHotEncoder preprocesser and processes the data.
    Then, it saves the preprocesser into a specified ubication and returns the data processed.
    '''

    encoder = OneHotEncoder(sparse=sparse_ohe)
    X_train_onehot = encoder.fit_transform(X_train)
    X_test_onehot = encoder.transform(X_test)
    pickle.dump(encoder, open(ohe_ubication, 'wb'))
    return X_train_onehot, X_test_onehot

#---------------------------------------------------------------------------------------------------------------

def GS_random_forest(X_train, y_train):

    '''
    This functions makes a GridSearch to calculate the best parameters for the RandomForestClassifier model.
    Then, returns these parameters.
    '''
    
    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt', n_estimators=50, oob_score = True)
    param_grid = { 
        'n_estimators': [200, 500, 700, 800],
        'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    GS_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    GS_rfc.fit(X_train, y_train)
    print(GS_rfc.best_params_)
    return GS_rfc.best_params_

#---------------------------------------------------------------------------------------------------------------

def rfc_train(X_train, y_train, params, n_splits_kfold = 5, shuffle_kfold = True, random_state_kfold = 42):

    '''
    This function trains a RandomForestClassifier model with the parameters specified and returns the accuracy scores 
    for each KFold to ensure it will work properly.
    '''
  
    kf = KFold(n_splits=n_splits_kfold, shuffle=shuffle_kfold, random_state=random_state_kfold)
    accuracy_scores_forest = []
    for train, val in kf.split(X_train):
        rfc = RandomForestClassifier(max_features = params['max_features'],
                                    n_estimators=params['n_estimators'], 
                                    max_depth=params['max_depth']
                                    )
        rfc = rfc.fit(X_train[train],y_train[train])
        
        # true positives, false negatives, false positives, true negatives
        tp, fn, fp, tn = confusion_matrix(y_train[val],rfc.predict(X_train[val])).ravel()
        r_accuracy = (tn + tp)/(tn+tp+fn+fp)
        r_prec = tp/(tp+fp)
        r_rec = tp/(tp+fn)
        r_fpr = fp/(fp+tn)
        r_f1 = 2*(r_prec)*r_rec/(r_prec+r_rec)
        accuracy_scores_forest.append((r_accuracy,r_prec,r_rec,r_fpr,r_f1))

    acc_rforest = np.mean(accuracy_scores_forest, axis=0)
    print(acc_rforest)

    return acc_rforest

#---------------------------------------------------------------------------------------------------------------

def rfc_test(X_test, y_test, X_train, y_train, params, model_ubication):

    '''
    This function trains a RandomForestClassifier with the specified parameters and saves the model.
    It will return the accuracy obtained in the test.
    '''

    rfc = RandomForestClassifier(max_features = params['max_features'],
                                n_estimators=params['n_estimators'], 
                                max_depth=params['max_depth']
                                )
    rfc = rfc.fit(X_train, y_train)
    tp, fn, fp, tn = confusion_matrix(y_test, rfc.predict(X_test)).ravel()
    accuracy = (tn+tp)/(fp+fn+tp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    fpr = fp/(fp+tn)
    f1 = 2*precision*recall/(precision + recall)
    pickle.dump(rfc, open(model_ubication, 'wb'))
    print(accuracy)
    return accuracy

#---------------------------------------------------------------------------------------------------------------

def prediction_vs_test(X_test, y_test, models_folder):

    '''
    This function prints a dataframe with the test labels and the results of the prediction in two columns.
    Also, it will return the prediction.
    '''
    loaded_model = pickle.load(open(models_folder, 'rb'))
    y_pred = loaded_model.predict(X_test)
    df = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
    print(df)
    return y_pred

#---------------------------------------------------------------------------------------------------------------

def app_prediction(url, ohe_folder, models_folder):

    '''This function calls other functions to transform the url into a usefull format for the prediction model.'''

    ip = ip_url(url)
    length = length_url(url)
    tiny = tiny_url(url)
    At = at_sign(url)
    double_slash = double_slash_url(url)
    prefix_suffix = prefix_suffix_url(url)
    sub_domain = subdomain_multisubdimain(url)
    token = https_token(url)
    
    data_list = [ip, length, tiny, At, double_slash, prefix_suffix, sub_domain, token]
    data = np.array([data_list])
    phishing_count = data_list.count(2)
    legitimate = data_list.count(1)
    suspicious = data_list.count(0)*0.5

    loaded_ohe = pickle.load(open(ohe_folder, 'rb'))
    data_encoded = loaded_ohe.transform(data)

    prediction = predict(data_encoded, models_folder)

    if prediction == 1:
        result = 'legitimate'
    else:
        result = 'phishing'

    return result

#---------------------------------------------------------------------------------------------------------------

def predict(data, models_folder):

    '''
    This function predicts wether it is a phising url or not and returns the result.
    '''
    loaded_model = pickle.load(open(models_folder, 'rb'))
    pred = loaded_model.predict(data)

    if pred == 1:
        result = 'legitime'
    else:
        result = 'phishing'

    return result

#---------------------------------------------------------------------------------------------------------------

def plot_confusion_matrix(y_test, models_folder, X_test):

    '''
    This function shows a confusion matrix to see the results visually.
    '''

    loaded_model = pickle.load(open(models_folder, 'rb'))
    y_pred = loaded_model.predict(X_test)
    CM = confusion_matrix(y_test, y_pred)
    ax = plt.axes()
    sns.heatmap(CM, annot=True, annot_kws={'size':10}, ax=ax )
    ax.set_title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#---------------------------------------------------------------------------------------------------------------

def subdomain_multisubdimain(url):

    '''
    This funtion returns 1 if the URL has one subdomain, 0 if the URL has two subdomains and 2 if it has more.
    '''
    pattern = re.compile(r'https?://([A-Za-z_0-9.-]+).*')
    url_re = pattern.findall(url)
    if 'www' in url:
        url_re_splited = url_re[0].split('.')
        subdomains = url_re_splited[2:]
        if len(subdomains) == 1:
            n = 1
        elif len(subdomains) == 2:
            n = 0
        else:
            n = 2

    elif 'http' not in url:
        n = 0

    else:
        pattern = re.compile(r'https?://([A-Za-z_0-9.-]+).*')
        url_re = pattern.findall(url)
        url_re_splited = url_re[0].split('.')
        subdomains = url_re_splited[2:]
        if len(subdomains) == 0:
            n = 1
        elif len(subdomains) == 1:
            n = 0
        else:
            n = 2
    return n

#---------------------------------------------------------------------------------------------------------------

def https_token(url):

    '''
    This function returns 2 if there are http tokens in the domain. 
    '''

    pattern = re.compile(r'(http)')
    token = pattern.findall(url)
    if len(token) <= 1:
        n = 1
    else:
        n = 2
    return n

#---------------------------------------------------------------------------------------------------------------

def tiny_url(url):

    '''
    This function returns 2 if a shorted url is a TinyURL and 1 if not.
    '''
    if len(os.path.split(url)[1]) < 10: 
        n = 1
    else:
        n = 2
    return n

#---------------------------------------------------------------------------------------------------------------

def double_slash_url(url):

    '''The existence of “//” within the URL path. If returns 1 is legitimate, if returns 2 is phishing'''

    if len(url.split('//')) > 2:
        n=2
    else:
        n=1
    return n

#---------------------------------------------------------------------------------------------------------------

def prefix_suffix_url(url):

    '''The existence of "-" whithin the URL path.If returns 1 is legitimate, if returns 2 is phishing'''

    if "-" in url:
        n=2
    else:
        n=1
    return n

#---------------------------------------------------------------------------------------------------------------

def ip_url(url):

    '''The existence of ip whithin the URL path.If returns 1 is legitimate, if returns 2 is phishing'''

    pattern = re.compile(r'(?:\d{1,3}\.)+(?:\d{1,3})') #busqueda de ruta con numeros
    if pattern.findall(url):
        n=2
    else:
        n=1
    return n

#---------------------------------------------------------------------------------------------------------------

def at_sign(url):

    '''The existence of "@" whithin the URL path.If returns 1 is legitimate, if returns 2 is phishing'''

    if '@' in url:
        n=2
    else:
        n=1
    return n

#---------------------------------------------------------------------------------------------------------------

def length_url(url):
    '''Long URL'''
    if len(url) < 54 :
        n=0
    elif len(url) >= 54 and len(url) <= 75:
        n=1
    elif len(url) >= 75 :
        n=2
    return n