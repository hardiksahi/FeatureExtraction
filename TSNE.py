import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import time

train = pd.read_csv("train_MNIST.csv") # data frame


target = train['label'].values
train = train.drop("label",axis=1)
Target = target[:6000]

X= train[:6000].values
del train
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(X_std)
X_50d = pca.transform(X_std)

#print("Hi")
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_50d) # 2 dimensions n lower space....

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
print(tsne_results.shape)
#print("Bye")


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2,3,4,5,6,7,8,9,10),
                        ('b', 'r', 'g','c','m','y','b','#800000','#33E6FF','#7A33FF')):
        rowIndex = np.where(Target == lab)[0]
        plt.scatter(tsne_results[rowIndex,0],
                    tsne_results[rowIndex,1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


