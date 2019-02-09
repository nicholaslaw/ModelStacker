from MLStacker import ModelStacker
stacker = ModelStacker()

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dtclf = DecisionTreeClassifier()
knnclf = KNeighborsClassifier()
svmclf = SVC()

stacker.add_base_model(dtclf)
stacker.add_base_model(knnclf)
stacker.add_base_model(svmclf)

from sklearn.linear_model import LogisticRegression
lgclf = LogisticRegression()
stacker.add_stacked_model(lgclf)

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
X_train = X[:120]
X_test = X[120:]
Y = iris.target
Y_train = Y[:120]
Y_test = Y[120:]
print("X shape: ", X.shape)
print("Y Shape: ", Y.shape)
stacker.fit(X_train, Y_train)
print("X test shape", X_test.shape)
print("Y test shape: ", Y_test.shape)
print(stacker.predict(X_test).shape)