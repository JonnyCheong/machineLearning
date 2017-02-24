#The code do the machine learning. Finally export .pkl file.
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
"""
The program using the data of dimensionality reduction to machine learning.
"""
data=np.loadtxt('bvh.csv',delimiter=',')#Load the bvh.csv file(data).
X=data[:,1:]#set the data at X.
y=data[:,0]#set the category(Label) of data.
print 'Start PCA...'
pca=PCA(n_components=2)#Define the multidimensional data to two-dimensional.
pca.fit(X)#Reducing the multi dimensional to two-dimensional
print pca.transform(X)
joblib.dump(pca.transform(X),'save/PCA.pkl')#Export the .pkl file.
print 'Export pkl done...'
X_train,X_test,y_train,y_test=train_test_split(pca.transform(X),y,test_size=0.3,random_state=42)
print 'Start Linear SVC...'
lsvc=SVC(kernel='linear')#Define choose using model(Linear SVC).
lsvc.fit(X_train,y_train)#Machine learning.
joblib.dump(lsvc,'save/linearSVC_PCA.pkl')#Export the .pkl file.
print 'Export pkl done...'
print 'Start SVC'
svc=SVC()#Define choose using model(SVC).
svc.fit(X_train,y_train)#Machine learning.
print 'Accuracy of SVC is ',svc.score(X_test,y_test)
joblib.dump(svc,'save/SVC_PCA.pkl')#Export the .pkl file.
print 'Export pkl done...'
print 'Start KNeighbors Classifier'
knn = KNeighborsClassifier()#Define choose using model(KNeighbors Classifier).
knn.fit(X_train,y_train)#Machine learning.
print 'Accuracy of KNeighbors Classifier is ',knn.score(X_test,y_test)
joblib.dump(knn,'save/knn_PCA.pkl')#Export the .pkl file.
print 'Export pkl done...'


