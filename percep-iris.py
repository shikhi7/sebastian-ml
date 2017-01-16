import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from percep1 import perceptron
from adaline import adaline
from matplotlib.colors import ListedColormap
# getting the data frame
df=pd.read_csv('https://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data', header=None)
# print(df.tail(n=10))

#select the flower variety. 
#we're going for two flower classification


y=df.iloc[0:100,4].values
# print(y)
y=np.where( y=='Iris-setosa',-1,1)

X=df.iloc[0:100,[0,2]].values
# print(X)
# print(y[49],y[50])
# we already know first 50 belong to setosa
#Plotting the dataframe.
plt.scatter(X[:50,0],X[:50,1],color="red", marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color="blue",marker='x',label="versicolor")
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


#Lets train the perceptron.
ppn = perceptron()
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Updates')
plt.tight_layout()
plt.show()

#Plotting decision boundaries.

def plot_decision_regions(X,y,classifier,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plotting the surface
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    #We get a grid from above. More like getting a graph paper.
    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)

plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

X_std=np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

#Now using Adaline model, lets plot.
ada=adaline(n_iter=15,eta=0.01)
ada.fit(X_std,y)
plot_decision_regions(X_std,y,classifier=ada)
plt.title('Adaline:Gradient')
plt.xlabel('Sepal (std)')
plt.ylabel('Petal (std)')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.tight_layout()
plt.show()

