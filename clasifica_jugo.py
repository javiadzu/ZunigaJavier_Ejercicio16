import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sklearn.tree
import sklearn.metrics
import matplotlib.pyplot as plt

#Este método de boobstrapping lo tomé de https://gist.github.com/aflaxman/6871948
def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample



# Carga datos
data = pd.read_csv('OJ.csv')

# Remueve datos que no se van a utilizar
data = data.drop(['Store7', 'PctDiscCH','PctDiscMM'],axis=1)

# Crea un nuevo array que sera el target, 0 si MM, 1 si CH
purchasebin = np.ones(len(data), dtype=int)
ii = np.array(data['Purchase']=='MM')
purchasebin[ii] = 0


data['Target'] = purchasebin

# Borra la columna Purchase
data = data.drop(['Purchase'],axis=1)

# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
predictors.remove('Unnamed: 0')

x_train, x_test, y_train, y_test = train_test_split(data, purchasebin, train_size=0.5)

xtrain= np.array(x_train)
xtest= np.array(x_test)
ytrain=np.array(y_train)


F1scoreM=np.zeros((100,10))


for j in range(100):
    x_tr=bootstrap_resample(xtrain)
    for i in range (10):
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=i+1)
        
        clf.fit(x_tr, ytrain)
        F1scoreM[j,i]=sklearn.metrics.f1_score(ytrain, clf.predict(x_tr))

prom=np.zeros(10)
for i in range(10):
    prom[i]=np.average(F1scoreM[i])
    plt.plot(i,prom[i],'o')
plt.xlabel('max depth')
plt.ylabel('average F1_score')
plt.savefig('AverageF1_score')