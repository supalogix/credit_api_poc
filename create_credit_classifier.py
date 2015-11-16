import pandas as pd

###
### Load Data Set
###
path='cs-training.csv'
df=pd.read_csv(path, 
               sep=',',
               header=0)
data = df.drop(df.columns[0], axis=1)

# DROP ROWS WITH n/a
data = data.dropna()

###
### CONVERT DATA INTO LIST OF DICTS RECORDS
###
data = data.to_dict(orient='records')

###
### one-of-K ENCODING OF CATEGORICAL FEATURES
###
from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
vec = DictVectorizer()
dataOneK = vec.fit_transform(data).toarray()
dataOneK = DataFrame(dataOneK,columns=vec.get_feature_names())
label_feature = vec.get_feature_names()

outcome_feature = ['SeriousDlqin2yrs']
label_feature.remove('SeriousDlqin2yrs')
outcome = dataOneK['SeriousDlqin2yrs']
dataOneK = dataOneK.drop('SeriousDlqin2yrs',axis=1)    




###
### Generate Training and Testing Set 
###
from sklearn import cross_validation

"""
    X_1: independent variables for first data set
    Y_1: dependent (target) variable for first data set
    X_2: independent variables for the second data set
    Y_2: dependent (target) variable for the second data set
"""
X_1, X_2, Y_1, Y_2 = cross_validation.train_test_split(
    dataOneK, outcome, test_size=0.5, random_state=0)
    
    
    

    
###
### Define Classifier
###                             
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


###
### Train Random Forest on (X1,Y1) and Validate on (X2,Y2)
###                              
clf.fit(X_1,Y_1)
score = clf.score(X_2, Y_2)
print "accuracy: {0}".format(score.mean())


###
### Print Confusion Matrix
###

output = clf.predict(X_2)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(output, Y_2)
print matrix





###
### Save Classifier
###
from sklearn.externals import joblib
joblib.dump(clf, 'model/nb.pkl')


