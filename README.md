# ModelStacker
ModelStacker implements the stacking of machine learning models and very often, the stacked model is able to perform better than any of its base models. This technique is said to be the most effective when there are vast differences present amongst the base models. More information on this concepts can be found at:

1. [A Kaggler's Guide to Model Stacking in Practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)

2. [Stacking Models for Improved Predictions](https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html)

## Dependencies
Currently, ModelStacker depends on:
- numpy==1.14.0
- pandas==0.22.0

## Installation
```
pip install MLStacker
```

## Usage
### Initalize ModelStacker
```
from MLStacker import ModelStacker
stacker = ModelStacker()
```

### Initialize and Add Base Models
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dtclf = DecisionTreeClassifier()
knnclf = KNeighborsClassifier()
svmclf = SVC()

stacker.add_base_model(dtclf)
stacker.add_base_model(knnclf)
stacker.add_base_model(svmclf)
```

### Initalize and Add Stacked Model
```
from sklearn.linear_model import LogisticRegression
lgclf = LogisticRegression()
stacker.add_stacked_model(lgclf)
```

### Fitting and Predicting
```
stacker.fit(X_train, y_train) # X_train and y_train belongs to training set
predictions = stacker.predict(X_test)
```