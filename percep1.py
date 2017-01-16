import numpy as np

class perceptron(object):
    """ eta:learning rate

n_iter: iterations """


    def __init__(self,eta=0.1,n_iter=10,randoms=1):
        self.eta=eta
        self.n_iter=n_iter
        self.randoms=randoms


    def fit(self,X,y):
        """
        X: nsamples X nfeatures
        y: nsamples X 1
        """
        # self.w_ = np.zeroes(1+X.shape[1])
        # we dont want all the weights to be zero 
        rgen=np.random.RandomState(self.randoms)
        # normal distribution, center=0,deviation=0.01
        #print(rgen)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors=0
            for xi,target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                """Because thats additional weight. Soit takes 1 as the multiplier"""
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    def predict(self,X):
        return np.where(self.net_input(X)>=0.0,1,-1)
