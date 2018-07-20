import numpy as np
import pandas as pd

train = pd.read_csv("train_MNIST.csv") # data frame


target = train['label'].values
train = train.drop("label",axis=1)
#print(target.head())

#print(train.shape)
#print(train.columns)

# Applying PCA
from sklearn.preprocessing import StandardScaler
X = train.values # This is now a matrix...
X_std = StandardScaler().fit_transform(X)

#print(np.mean(X_std, axis = 0))
#print(np.std(X_std, axis = 0))

cov_matrix = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

#print(eig_vals.shape)
#print(eig_vecs.shape)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(eig_vals.shape[0])]
print("eig_pairs")
# Sort eigen values 
eig_pairs.sort(key = lambda x: x[0], reverse = True)
print("Sorted")
# Total variance
tot = np.sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse = True)]

cum_var_exp = np.cumsum(var_exp)

from matplotlib import pyplot as plt
'''print("Scree plot")

xValues = [i+1 for i in range(1,len(var_exp)+1)]
plt.plot(xValues, var_exp, color = 'green', marker='o')
plt.plot(xValues, cum_var_exp, color = 'red', marker='o')
plt.title("Scree plot")
plt.xlabel("Principal component")
plt.ylabel("Proportion of variance.")

plt.legend(["PC proportion", "Cumulative variance"])'''


'''print("Displaying first 70 numbers...")
plt.figure(figsize=(14,12))
for digit_num in range(0,70):
    plt.subplot(7,10,digit_num+1)
    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()'''


# Use only first 2 components.....
# Way 1: Using self PCA

PStarMatrix = np.hstack((eig_pairs[0][1].reshape(784,1), eig_pairs[1][1].reshape(784,1)))
print(PStarMatrix.shape)

# Way 2: Using inbuild PCA

# Transform X_std using projection matrix

'''ProjX = np.dot(X_std, PStarMatrix)
print("XSTAR shape", ProjX.shape) # only 2 dimesnions.

# Plpot 2 dimensions ......



with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2,3,4,5,6,7,8,9,10),
                        ('b', 'r', 'g','c','m','y','b','#800000','#33E6FF','#7A33FF')):
        rowIndex = np.where(target == lab)[0]
        plt.scatter(ProjX[rowIndex,0],
                    ProjX[rowIndex,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()'''
    
## Plotting only top 6000 entries in scatterplot in plotly using scikit PCA
del X
X= train[:6000].values
del train

X_std = StandardScaler().fit_transform(X)


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(X_std)
X_5d = pca.transform(X_std) # projected in 5 dimensions specified by PCA

Target = target[:6000]


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2,3,4,5,6,7,8,9,10),
                        ('b', 'r', 'g','c','m','y','b','#800000','#33E6FF','#7A33FF')):
        rowIndex = np.where(Target == lab)[0]
        plt.scatter(X_5d[rowIndex,0],
                    X_5d[rowIndex,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()


