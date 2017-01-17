import numpy as np
import pandas as pd
#The following is for finding the best parameters, and for logistic regression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
###Let us load the data from cleaned csv file
df=pd.read_csv('./movie_data.csv')

###test if working
# print(df)

###Let us perform a test train split

X_train=df.loc[:25000,'review'].values
y_train=df.loc[:25000,'sentiment'].values
X_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values

###some essential definitions
porter=PorterStemmer()
def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
stop=stopwords.words('english')
###Finding the best parameters.

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid=[{'vect__ngram_range':[(1,1)],
             'vect__stop_words':[stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'clf__penalty':['l1','l2'],
             'clf__C':[1,10,100]},
            {'vect__ngram_range':[(1,1)],
             'vect__stop_words':[stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'vect__use_idf':[False],
             'vect__norm':[None],
             'clf__penalty':['l1','l2'],
             'clf__C':[1,10,100]}]
#The first set uses tf-idf method. Second one doesn't.
#tokenizer: Simply return space separated words.
#tokenizer_porter: Return the root words, using Porter Stemming from nltk.
#k fold cross validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation

lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
#pipeline has (key,value) pairs, where key is the string that we want to call the step.
#value is an estimator object
#Parameters of the estimators in the pipeline can be accessed using the <estimator>__<parameter> syntax
#thus when we write vect__norm et al, they are for the step with tf-idf transforms
#And when we write clf__C, they change the C parameters sent in Logistic Regression Classifier
#This step is sexy, letting us do so much, in so less
#Hail GridSearchCV
#We have a fit method and a transform method[Above]

gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
gs_lr_tfidf.fit(X_train,y_train)

###fitting with training data
gs_lr_tfidf.fit(X_train,y_train)

###Print the best parameters
print(gs_lr_tfidf.best_params_)
###The best score
print(gs_lr_tfidf.best_score_)

###Initializing the best estimator object
clf=gs_lr_tfidf.best_estimator_

###Printing the test score, running on the test data
print(clf.score(X_test,y_test))

