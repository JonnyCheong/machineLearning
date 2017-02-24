import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.externals import joblib
print 'Load the .csv file...'
data=np.loadtxt('bvh.csv',delimiter=',')#Load the bvh.csv file(data).
print 'Done.'
X=data[:,1:]#set the data at X.
y=data[:,0]#set the category(Label) of data.
print 'Start spliting data...'
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)#Divide the data in testing and training.
print 'Done.'
print 'Start Linear SVC...'
lsvc=LinearSVC()#Define choose using model(Linear SVC).
lsvc.fit(X_train,y_train)#Machine learning.
print 'Accuracy of SVC is ',lsvc.score(X_test,y_test)# Testing accuracy of trained machine.
print 'Start Export .pkl file...'
oblib.dump(lsvc,'save/linearSVC.pkl')#Export the .pkl file.
print 'Export pkl done.'
#Principal components analysis
print 'Start PCA...'
pca=PCA(n_components=2)#Define the multidimensional data to two-dimensional.
pca.fit(X)#Reducing the multi dimensional to two-dimensional
print pca.transform(X) #Print out the result
print 'Start Export .pkl file...'
oblib.dump(pca.transform(X),'save/PCA.pkl')#Export the .pkl file.
print 'Export pkl done.'