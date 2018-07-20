import numpy as np
import pandas as pd

train = pd.read_csv("train_MNIST.csv") # data frame


target = train['label']
train = train.drop("label",axis=1)

y = target[:6000]
X= train[:6000].values

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
X = selector.fit_transform(X)
print("Dimension after select", X.shape)

classCount = np.unique(y).shape[0]
print("classCount",classCount)


# Standardize data
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(X)
X = pca.transform(X)

dimensions = X.shape[1]
print("dimensions",dimensions)

def computeClassMeansList(X,y, classCount):
    classMeanList = []
    for cl in range(1,classCount+1): #1,2,3
        meanVectorClass = np.mean(X[y ==cl], axis =0)
        classMeanList.append(meanVectorClass)
    return classMeanList

def computeWithinClassMatrix(X, y,classMeanList, classCount, dimensions):
    withinClassMatrix = np.zeros((dimensions, dimensions))
    
    for cl, meanV in zip(range(1,classCount+1), classMeanList):
        matrix = np.zeros((dimensions, dimensions))
        for row in X[y == cl]:
            row = row.reshape((dimensions,1))
            meanV = meanV.reshape((dimensions, 1))
            m = np.dot(row-meanV, (row-meanV).transpose())
            withinClassMatrix = np.add(withinClassMatrix, m)
        withinClassMatrix = np.add(withinClassMatrix, matrix)
    return withinClassMatrix

def computeBetweenClassMatrix(X, y, classMeanList, dimensions):
    inBetweenMatrix = np.zeros((dimensions, dimensions))
    overallMeanvector = np.mean(X, axis = 0).reshape((dimensions,1)) # columns
    for index, classMeanV in enumerate(classMeanList): 
        rowsInClass = X[y==(index+1)].shape[0]
        classMeanV = classMeanV.reshape((dimensions,1)) # as a column
        inBetweenMatrix+=np.dot(classMeanV-overallMeanvector, (classMeanV-overallMeanvector).transpose())*rowsInClass
    return inBetweenMatrix
    
'''classMeanList = computeClassMeansList(X,y, classCount)
withinClassMatrix = computeWithinClassMatrix(X, y, classMeanList, classCount, dimensions)
betweenClassMatrix = computeBetweenClassMatrix(X,y,classMeanList,dimensions)

# perform EVD of within inverse, between
eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(withinClassMatrix), betweenClassMatrix))

# create tuples of eigvalues and vectors
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(0, eig_vals.shape[0])]

# Sort basis of eigen values
eig_pairs.sort(key = lambda x: x[0], reverse = True)'''

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 5)

lda_X = lda.fit_transform(X, y)
print(lda_X.shape)

import matplotlib.pyplot as plt
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2,3,4,5,6,7,8,9,10),
                        ('b', 'r', 'g','c','m','y','b','#800000','#33E6FF','#7A33FF')):
        rowIndex = np.where(y == lab)[0]
        plt.scatter(lda_X[rowIndex,0],
                    lda_X[rowIndex,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()

