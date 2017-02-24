from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
X_pca=joblib.load('save/PCA.pkl')
lsvc = joblib.load('save/linearSVC_PCA.pkl')
print 'Load the .csv file...'
data=np.loadtxt('bvh.csv',delimiter=',')
X=data[:,1:]
y=data[:,0]
ipca = IncrementalPCA(n_components=2, batch_size=10)
y_p = lsvc.predict(X_pca)
print y_p
name=['zero','one','two','three','four','five']
colors = ['navy', 'turquoise', 'darkorange','red','green','yellow']
print lsvc
for X_transformed, title in [(X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2,3,4,5],name ):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                    color=color, lw=0.5, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-200,200,-100,100])

for X_transformed, title in [(X_pca, "After Linear svc")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2,3,4,5],name ):
        plt.scatter(X_transformed[y_p == i, 0], X_transformed[y_p == i, 1],
                    color=color, lw=0.5, label=target_name)
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-200,200,-100,100])
plt.show()
