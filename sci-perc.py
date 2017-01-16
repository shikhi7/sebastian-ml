# First steps with scikit learn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from plot_boundaries import plot_decision_regions

import matplotlib.pyplot as plt

iris=datasets.load_iris()
# print(iris)
X=iris.data[:,[2,3]]
y=iris.target
# print(np.unique(y))
#Splitting it into test and train data.
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

#Training perceptron

ppn=Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(X_train_std,y_train)
y_pred=ppn.predict(X_test_std)
print((y_test!=y_pred).sum())

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,y=y_combined,classifier=ppn,test_idx=range(105,150))
plt.xlabel('Petal length[std]')
plt.ylabel('Petal width[std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()



lr=LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))
plt.xlabel('Petal length [std]')
plt.ylabel('Petal width [std]')
plt.legend(loc='upper left')
plt.show()
