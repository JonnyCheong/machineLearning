import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
data=np.loadtxt('Share/bvh.csv',delimiter=',')
X=data[:,1:]
y=data[:,0]
pca=PCA(n_components=2)
pca.fit(X)
xpca=pca.transform(X)
print xpca
X_train,X_test,y_train,y_test=train_test_split(pca.transform(X),y,test_size=0.3,random_state=42)
clf = LinearSVC()
clf.fit(X_train, y_train)
print clf.score(X_test,y_test)
h = .02  
x_min, x_max = xpca[:, 0].min() - 1, xpca[:, 0].max() + 1
y_min, y_max = xpca[:, 1].min() - 1, xpca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired,alpha=0.5)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,)
plt.title('Linear SVC')
plt.axis('tight')
plt.show()