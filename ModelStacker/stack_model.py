import numpy as np
import pandas as pd
import copy
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html
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

    def fit(self, X, Y, shuffle=True, seed=0, folds=5):
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
        if shuffle:
            combined = np.hstack((X, Y))
            np.random.seed(seed)
            np.random.shuffle(combined)
            X = combined[:, :-1]
            Y = combined[:, -1]
            del combined
        # Validation splits
        X_split = np.array(np.array_split(X, folds))
        Y_split = np.array(np.array_split(Y, folds))
        assert len(X_split) == len(Y_split)
        # Stacking Starts
        index_lst = list(range(len(X_split)))
        all_mod_preds = []
        for key, mod in list(self.base_models.items()):
            mod_pred = []
            for idx, X_chunk in enumerate(X_split):
                temp_indices = index_lst.copy()
                temp_indices.remove(idx)
                temp_mod = copy.deepcopy(mod)
                temp_mod.fit(X_split[temp_indices], Y_split[temp_indices])
                mod_pred.extend(list(temp_mod.predict(X_chunk)))
            all_mod_preds.append(mod_pred)
            # Fit model to all training data
            mod.fit(X, Y)
            self.base_models[key] = mod
        # Concatenating Features Generated
        initial_col = X.shape[1]
        for pred in all_mod_preds:
            X = np.hstack((X, np.array(pred)))
        then = X.shape[1] - initial_col
        assert then == len(self.base_models)
        return X, Y
        