import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
import joblib

def load_data_set():
    path='cs-training.csv'
    df=pd.read_csv( path, sep=',', header=0 )
    data = df.drop( df.columns[0], axis=1)

    return data

def get_features(data):
    ###
    ### Drop rows with missing column data
    ###

    data = data.dropna()

    ###
    ### Convert Data Into List Of Dict Records
    ###

    data = data.to_dict(orient='records')

    ###
    ### Seperate Target and Outcome Features
    ###

    vec = DictVectorizer()

    df_data = vec.fit_transform(data).toarray()
    feature_names = vec.get_feature_names()
    df_data = DataFrame(
        df_data,
        columns=feature_names)

    outcome_feature = df_data['SeriousDlqin2yrs']
    target_features = df_data.drop('SeriousDlqin2yrs', axis=1)

    return outcome_feature, target_features



data = load_data_set()
outcome_feature, target_features = get_features(data)

###
### Generate Training and Testing Set 
###

"""
    X_1: independent variables for first data set
    Y_1: dependent (target) variable for first data set
    X_2: independent variables for the second data set
    Y_2: dependent (target) variable for the second data set
"""
X_1, X_2, Y_1, Y_2 = train_test_split(
    target_features, 
    outcome_feature, 
    test_size=0.5, 
    random_state=0)
    
###
### Define Classifier
###                             

clf = GaussianNB()

###
### Train Classifier on (X1,Y1) and Validate on (X2,Y2)
###                              

clf.fit(X_1,Y_1)
score = clf.score(X_2, Y_2)
print("accuracy: {0}".format(score.mean()))

###
### Print Confusion Matrix
###

output = clf.predict(X_2)

matrix = confusion_matrix(output, Y_2)
print(matrix)

###
### Save Classifier
###
joblib.dump(clf, 'model/nb.pkl')