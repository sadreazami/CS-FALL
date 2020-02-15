
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from PIL import Image
import os
from scipy import ndimage, misc
from keras.utils import np_utils


path = 'C:/Users/Hamidreza/Desktop/Dipayan/Result/0/C_imagen/'

SSS=206
dataa = np.zeros((SSS,129,138))
zz=[]

for i in range(SSS):    
    zz.append(str(i+1)+".bmp")

for ii, imagee in enumerate(zz):
    path2 = os.path.join(path, imagee)
    image2 = ndimage.imread(path2)
    image2=image2.astype(np.float64)
    dataa[ii,:,:]=image2/255

import csv
with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     label=np.squeeze(label) 
     
#label=np.repeat(label,10)

m=4 

kf=KFold(5, random_state=None, shuffle=False)
kf.get_n_splits(dataa)
k=0
for train_index, test_index in kf.split(dataa):
    X_train, X_test = dataa[train_index], dataa[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    if k==m:
       break 
    k=k+1
       
n_samples = 129*138
data = X_train.reshape([X_train.shape[0],n_samples, -1])
data_test = X_test.reshape([X_test.shape[0],n_samples, -1])
data=np.squeeze(data) 
data_test=np.squeeze(data_test) 

##############################GSVM
from sklearn import svm
classifier = svm.SVC(C=1, gamma=0.001)
classifier.fit(data, y_train)
expected = y_test
predicted = classifier.predict(data_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

##############################LSVM
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0, penalty='l2',)
clf.fit(data, y_train)
predicted2 = clf.predict(data_test)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))

##############################KNN
from sklearn.neighbors import KNeighborsClassifier

for k in np.arange(1, 2, 1):
     model = KNeighborsClassifier(n_neighbors=k)
     model.fit(data, y_train)
     predictions1 = model.predict(data_test)
     print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_test, predictions1)))
     print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions1)) 

###################################Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini")
clf_gini.fit(data, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(data, y_train)

pred1 = clf_gini.predict(data_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf_gini, metrics.classification_report(expected, pred1)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, pred1))

pred2 = clf_entropy.predict(data_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf_entropy, metrics.classification_report(expected, pred2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, pred2))

#######################################Bayes
from sklearn.naive_bayes import GaussianNB

clf0=GaussianNB()
clf0.fit(data, y_train)
predic2 = clf0.predict(data_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf0, metrics.classification_report(expected, predic2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predic2))

#################################LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf3 = LinearDiscriminantAnalysis( solver='svd')
clf3.fit(data, y_train)

predi = clf3.predict(data_test)
print("Classification report for classifier %s:\n%s\n"
      % (clf3, metrics.classification_report(expected, predi)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predi))