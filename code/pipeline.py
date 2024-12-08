import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import jupyterthemes as jt
from jupyterthemes import jtplot
from jupyterthemes import get_themes

from imblearn.under_sampling import RandomUnderSampler

import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import sweetviz as sv
#from sklearn.preprocessing import StandardScaler
#from sklearn.utils import resample
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder


#################################
#make sure to run pip install jupyterthemes on your local comupter
#################################


def preprocess_data(df, target, columns_to_drop):
    """
    got rid of inf values and set null values equal to the mean 
    """
    df = df
    dff = df.replace([np.inf, -np.inf], np.nan)
    # makes sure all the columns that run the mean are numeric 
    numeric_cols = dff.select_dtypes(include=[np.number]).columns
    dff[numeric_cols] = dff[numeric_cols].fillna(dff[numeric_cols].mean())

    #drops columns that are part of the params of the function 
    X = dff.drop(columns=[target] + columns_to_drop)
    y = dff[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Instantiate the RandomUnderSampler instance
    rus = RandomUnderSampler(random_state=42)

    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    #creates a list for all the data 
    train_test_res = [X_train, X_test, y_train, y_test, X_resampled, y_resampled]
    
    return dff, train_test_res
    
def run_model(new_df, train_test_res):
    """
    train_test_res[0] = X_train 
    train_test_res[1] = X_test
    train_test_res[2] = y_train
    train_test_res[3] = y_test
    train_test_res[4] = X_resampled
    train_test_res[5] = y_resampled 
    """

    # create empty lists for the heat map at the end 

    model_names = []
    predictions = []

    # Initialize and fit the models
    
    #svc 
    clfr = SVC(kernel="rbf", random_state=42)
    clfr.fit(train_test_res[0], train_test_res[2])
    svc_pred = clfr.predict(train_test_res[1])
    print("SVC")
    print(classification_report(train_test_res[3], svc_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(clfr)
    predictions.append(svc_pred)
    
    #svc undersampled 
    clfr_svc_und = SVC(kernel="rbf", random_state=42)
    clfr_svc_und.fit(train_test_res[4], train_test_res[5])
    svc_und_pred = clfr_svc_und.predict(train_test_res[1])
    print("SVC undersampled")
    print(classification_report(train_test_res[3], svc_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(clfr_svc_und)
    predictions.append(svc_und_pred)


    # random forest
    clfr_rf = RandomForestClassifier(n_estimators=500, random_state = 42)
    clfr_rf.fit(train_test_res[0], train_test_res[2])
    rf_pred = clfr.predict(train_test_res[1])
    print("random forest")
    print(classification_report(train_test_res[3], rf_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(clfr_rf)
    predictions.append(rf_pred)
    
    # random forest undersampled 
    clfr_rf_und = RandomForestClassifier(n_estimators=500, random_state = 42)
    clfr_rf_und.fit(train_test_res[4], train_test_res[5])
    rf_und_pred = clfr_rf_und.predict(train_test_res[1])
    print("random forest undersampled")
    print(classification_report(train_test_res[3], rf_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(clfr_rf_und)
    predictions.append(rf_und_pred)
    
    #log reggession 
    logreg = LogisticRegression(random_state = 42)
    logreg.fit(train_test_res[0], train_test_res[2])
    log_pred = logreg.predict(train_test_res[1])
    print("log regg")
    print(classification_report(train_test_res[3], log_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(logreg)
    predictions.append(log_pred)

    #log reggession 
    logreg_und = LogisticRegression(random_state = 42)
    logreg_und.fit(train_test_res[4], train_test_res[5])
    log_und_pred = logreg_und.predict(train_test_res[1])
    print("log undersampled")
    print(classification_report(train_test_res[3], log_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(logreg_und)
    predictions.append(log_und_pred)

    #knn 
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(train_test_res[0], train_test_res[2])
    knn_pred = knn.predict(train_test_res[1])
    print("KNN")
    print(classification_report(train_test_res[3], knn_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(knn)
    predictions.append(knn_pred)

    #knn undersampled 
    knn_und = KNeighborsClassifier(n_neighbors = 5)
    knn_und.fit(train_test_res[4], train_test_res[5])
    knn_und_pred = knn_und.predict(train_test_res[1])
    print("KNN undersampled")
    print(classification_report(train_test_res[3], knn_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(knn_und)
    predictions.append(knn_und_pred)

    #decision tree 
    dectree = DecisionTreeClassifier(random_state=42)
    dectree.fit(train_test_res[0], train_test_res[2])
    dectree_pred = dectree.predict(train_test_res[1])
    print("DecisionTree")
    print(classification_report(train_test_res[3], dectree_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(dectree)
    predictions.append(dectree_pred)


    #decision tree undersampled
    dectree_und = DecisionTreeClassifier(random_state=42)
    dectree_und.fit(train_test_res[4], train_test_res[5])
    dectree_und_pred = dectree_und.predict(train_test_res[1])
    print("DecisionTree undersampled")
    print(classification_report(train_test_res[3], dectree_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    model_names.append(dectree_und)
    predictions.append(dectree_und_pred)

    # code below pretty much copyed then edited from the link https://github.com/BrunaClo/Star-Classification/blob/main/Star-Classification.ipynb

    # loop thru all the models and append the scores to each of the lists 
    # create empty lists for accuracy, precision, recall and f1
    a, p, r, f = [], [], [], []
    
    # Loop through the indices of the model_names and predictions
    for i in range(len(model_names)):
        a.append(accuracy_score(train_test_res[3], predictions[i]))
        p.append(precision_score(train_test_res[3], predictions[i]))
        r.append(recall_score(train_test_res[3], predictions[i]))
        f.append(f1_score(train_test_res[3], predictions[i]))
    
    # Create a DataFrame with the scores
    scores = pd.DataFrame([a, p, r, f], columns=['SVC', 'SVC (undersampled)', 'RF', 'RF (undersampled)', 'LogReg', 
                                                  'LogReg (undersampled)', 'KNN', 'KNN (undersampled)', 
                                                  'DecisionTree', 'DecisionTree (undersampled)']).T
    scores.columns = ['Accuracy', 'Precision', 'Recall', 'F1']
    scores = scores.sort_values(by='F1', ascending=False)
    
    # Plot the heatmap
    jtplot.style(figsize=(10, 10))
    sn.heatmap(scores, annot=True, cmap='coolwarm', fmt='.4f', linewidths=0)
    plt.xlabel('Scores')
    plt.ylabel('Models')
    plt.grid(False)
    plt.show()

