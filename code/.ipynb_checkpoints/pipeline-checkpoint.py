import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import sweetviz as sv
#from sklearn.preprocessing import StandardScaler
#from sklearn.utils import resample
def preprocess_data(df, target, columns_to_drop):
    """
    will fill null values as the mean and 
    split into training and testing sets. need to define Y column
    also sets infinte values to the 
    """
    df = df
    dff = df.replace([np.inf, -np.inf], np.nan)
    # makes sure all the columns that run the mean are numeric 
    numeric_cols = dff.select_dtypes(include=[np.number]).columns
    dff[numeric_cols] = dff[numeric_cols].fillna(dff[numeric_cols].mean())
    
    X = dff.drop(columns=[target] + columns_to_drop)
    y = dff[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    # Instantiate the RandomUnderSampler instance
    rus = RandomUnderSampler(random_state=42)

    # Fit the data to the model
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

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

    # Initialize and fit the models
    
    #svc 
    clfr = SVC(kernel="rbf", random_state=42)
    clfr.fit(train_test_res[0], train_test_res[2])
    svc_pred = clfr.predict(train_test_res[1])
    print("SVC")
    print(classification_report(train_test_res[3], svc_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    
    #svc resampled underfitting 
    clfr_svc_und = SVC(kernel="rbf", random_state=42)
    clfr_svc_und.fit(train_test_res[4], train_test_res[5])
    svc_und_pred = clfr_svc_und.predict(train_test_res[1])
    print("SVC undersampled")
    print(classification_report(train_test_res[3], svc_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")

    # random forest
    clfr_rf = RandomForestClassifier(n_estimators=500, random_state = 42)
    clfr_rf.fit(train_test_res[0], train_test_res[2])
    rf_pred = clfr.predict(train_test_res[1])
    print("random forest")
    print(classification_report(train_test_res[3], rf_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    
    # underfitting resampled random forest 
    clfr_rf_und = RandomForestClassifier(n_estimators=500, random_state = 42)
    clfr_rf_und.fit(train_test_res[4], train_test_res[5])
    rf_und_pred = clfr_rf_und.predict(train_test_res[1])
    print("random forest undersampled")
    print(classification_report(train_test_res[3], rf_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    
    #log reggession 
    logreg = LogisticRegression(random_state = 42)
    logreg.fit(train_test_res[0], train_test_res[2])
    log_pred = logreg.predict(train_test_res[1])
    print("log regg")
    print(classification_report(train_test_res[3], log_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")

    #log reggession 
    logreg_und = LogisticRegression(random_state = 42)
    logreg_und.fit(train_test_res[4], train_test_res[5])
    log_und_pred = logreg_und.predict(train_test_res[1])
    print("log undersampled")
    print(classification_report(train_test_res[3], log_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")

    #knn 
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(train_test_res[0], train_test_res[2])
    knn_pred = knn.predict(train_test_res[1])
    print("KNN")
    print(classification_report(train_test_res[3], knn_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")

    #knn undersampled 
    knn_und = KNeighborsClassifier(n_neighbors = 5)
    knn_und.fit(train_test_res[4], train_test_res[5])
    knn_und_pred = knn_und.predict(train_test_res[1])
    print("KNN undersampled")
    print(classification_report(train_test_res[3], knn_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")

    dectree = DecisionTreeClassifier(random_state=42)
    dectree.fit(train_test_res[0], train_test_res[2])
    dectree_pred = dectree.predict(train_test_res[1])
    print("DecisionTree")
    print(classification_report(train_test_res[3], dectree_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")

    dectree_und = DecisionTreeClassifier(random_state=42)
    dectree_und.fit(train_test_res[4], train_test_res[5])
    dectree_und_pred = dectree_und.predict(train_test_res[1])
    print("DecisionTree undersampled")
    print(classification_report(train_test_res[3], dectree_und_pred, labels = [1, 0]))
    print("------------------------------------------------------------------------")
    
    # Print the results
    #print("SVC")
    #pd.Series(clfr.feature_importances_, index = train_test_res[0].columns).sort_values().plot.barh()
    #print("------------------------------------------------------------------------")

    #print("SVC undersampled")
    #pd.Series(clfr_svc_und.feature_importances_, index = train_test_res[0].columns).sort_values().plot.barh()
    #print("------------------------------------------------------------------------")

    print("random forest")
    pd.Series(clfr_rf.feature_importances_, index = train_test_res[0].columns).sort_values().plot.barh()
    print("------------------------------------------------------------------------")

    print("random forest undersampled")
    pd.Series(clfr_rf_und.feature_importances_, index = train_test_res[0].columns).sort_values().plot.barh()
    print("------------------------------------------------------------------------")
    
    #print("log")
    #pd.Series(logreg.feature_importances_, index = train_test_res[0].columns).sort_values().plot.barh()
    #print("------------------------------------------------------------------------")
    #
    #print("log undersampled")
    #pd.Series(logreg_und.feature_importances_, index = train_test_res[0].columns).sort_values().plot.barh()


