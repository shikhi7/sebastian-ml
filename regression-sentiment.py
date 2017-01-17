import numpy as np
import pandas as pd
#The following is for finding the best parameters, and for logistic regression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#Let us load the data from cleaned csv file
df=pd.read_csv('./movie_data.csv')

#test if working
# print(df)

#Let us perform a test train split

X_train=df.loc[:25000,'review'].values
y_train=df.loc[:25000,'sentiment'].values
X_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values


