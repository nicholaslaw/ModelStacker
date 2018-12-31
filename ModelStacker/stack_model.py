import numpy as np
import pandas as pd
class ModelStacker:
    def __init__(self):
        self.base_models = {}
        self.stacked_model = None
    
    def add_base_model(self, model):
        """
        model: model object, preferably sklearn

        adds model objects for stacking
        """
        if not hasattr(model, "fit"):
            raise ValueError("Add method only takes in a model object which has fit method, such as models from sklearn or xgboost")
        temp_idx = len(self.base_models)
        self.base_models['model_' + str(temp_idx)] = model

    def fit(self, X, Y, shuffle=True, seed=0, folds=5, test=0.3):
        """
        X: pandas dataframe or numpy matrix
            Independent variables to be trained on
        Y: pandas series or numpy array
            Dependent variables to be trained on
        shuffle: boolean
            True if want to shuffle data before stacking
        folds: int
            cross validation folds for stacking
        test: float
            0 if does not want to split current dataset into training and test set

        returns new feature columns by stacking, number of feature columns will correspond to the number of models
        """
        if len(self.base_models) <= 1:
            raise Exception("Add more than 1 model for stacking to make sense")
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        if np.isnan(np.sum(X)):
            raise ValueError("X contains null values")
        if np.isnan(np.sum(Y)):
            raise ValueError("Y contains null vlaues")
        if len(X) != len(Y):
            raise ValueError("Number of training samples must be equal to the number of labels")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle should be a boolean")
        if not isinstance(seed, int):
            raise ValueError("seed should be an integer")
        if not isinstance(folds, int):
            raise ValueError("folds should be an integer")
        if folds < 2:
            raise ValueError("folds should be 2 or more")
        if not 0<=test<1:
            raise ValueError("test should be a value in [0, 1)")
        if shuffle:
            combined = np.hstack((X, Y))
            np.random.seed(seed)
            np.random.shuffle(combined)
            X = combined[:, :-1]
            Y = combined[:, -1]
            del combined
        if test > 0:
            test_samples = int(test * len(X))
            X_train = X[:test_samples, :]
            Y_train = Y[:test_samples, :]
            X_test = X[test_samples:, :]
            Y_test = Y[test_samples:, :]
            del X
            del Y
            X = X_train.copy()
            Y = Y_train.copy()
            del X_train
            del Y_train
        
