import numpy as np 
import pandas as pd  
from sklearn import tree
from sklearn import linear_model # of sklearn.lear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import math

X = [[0,0], [1,1]]
Y = [0,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
pred = clf.predict([0,1])

reg = linear_model.LinearRegression()
reg.fit([[0,0],[1,1],[2,2]], [0,1,2])
coef = reg.coef_ # 0.5 and 0.5

#can get coefficient w/ reg.coef_
#can get intercept w/ reg.intercept_
#reg.score with testing features and independents to get performance
#sklearn predictions are returne in an array so have to do [0][0]
#for coefficient [0][0] for intercept [0]


#plotting example
#plt.scatter(x,y)
#plt.plot(x, reg.predict(x), color='blue', linewidth=3)
#plt.xlabel("lala")
#plt.ylable("bla")
#plt.show()


#svm
clf2 = SVC()
clf2.fit(X, Y)
pred2 = clf2.predict([1,1])

train_features = [[1, 1], [2, 2], [3, 3], [4, 6], [5, 9]]
train_labels = [0, 0, 0, 1, 1]
test_feature = [6, 10]

#creating KNN algorithm 

def dis(x,y):
	sum_sq=0.0
 
	#add up the squared differences
	for i in range(len(x)):
		sum_sq += (x[i]-y[i])**2
 
	#take the square root of the result
	return (sum_sq ** 0.5)



def cn(features, labels, x):
	distances = []

	for index, value in enumerate(features):
		distances.append(dis(value, x))
		

	min_index = distances.index(min(distances))

	return labels[min_index]

#naive bayes
train_features = np.array([[1, 1], [2, 2], [3, 3], [4, 6], [5, 9]])
train_labels = np.array([0, 0, 0, 1, 1])
test_feature = np.array([6, 10])

clf3 = GaussianNB()
clf3.fit(train_features, train_labels)
#we've been using 'train' interchangeably with "fit"


#----

sleep = [5,6,7,8,10]
scores = [65,51,75,75,86]

mean_sleep = np.mean(sleep)
mean_scores = np.mean(scores)

normalized_sleep = sleep - mean_sleep
normalized_scores = scores - mean_scores
#-- screw understanding first principles linear regression solving

#-----











