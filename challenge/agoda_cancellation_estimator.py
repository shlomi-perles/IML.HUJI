from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np

# Aditiion imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from estimator_helper import *
from sklearn.model_selection import GridSearchCV


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, regressor_scale_x=True, regressor_scale_y=True,
                 classifier_scale_x=True) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.regressor_scale_x = regressor_scale_x
        self.regressor_scale_y = regressor_scale_y
        self.classifier_scale_x = classifier_scale_x
        # self.regressor_param_grid = {
        #     'n_estimators': [50, 100, 200, 500],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth': [None, 4, 7, 8, 100, 200, 300, 400],
        #     'criterion': ['squared_error']
        # }
        self.classifier_param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 8, 100],
            'criterion': ['gini', 'entropy']
        }
        # self.classifier_param_grid = {}
        self.classifier_param_grid = {'criterion': ['entropy'], 'max_depth': [100], 'max_features': ['auto'], 'n_estimators': [500]}
        self.regressor_param_grid = {
            'n_estimators': [600],
            'max_features': ['log2'],
            'max_depth': [100],
            'criterion': ['squared_error']
        }
        self.regressor_x_scaler = None
        self.classifier_x_scaler = None
        self.grid_search = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        y_classifier = y.y_classifier.to_numpy()


        self._fit_classifier(X, y_classifier)
        classifier_predict = self._predict_classifier(X)
        X = X.copy()
        X["canceled"] = classifier_predict

        cancel = []
        for index, row in X.iterrows():
            if row["canceled"] == 0:
                cancel.append(index)
        X.drop(cancel, axis=0, inplace=True)
        y.drop(cancel, axis=0, inplace=True)
        del X["canceled"]

        y_regressor = y.y_regressor.to_numpy()
        self._fit_regressor(X, y_regressor)

    def _fit_classifier(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        if self.classifier_scale_x:
            self.classifier_x_scaler = StandardScaler().fit(X)
            X_train_norm = self.classifier_x_scaler.transform(X)
            X = pd.DataFrame(X_train_norm, columns=X.columns)

        self.classifier_gridsearch = GridSearchCV(estimator=RandomForestClassifier(),
                                                  param_grid=self.classifier_param_grid,
                                                  scoring='r2')
        self.classifier_gridsearch.fit(X=X, y=y)

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        if self.regressor_scale_x:
            self.regressor_x_scaler = StandardScaler().fit(X)
            X_train_norm = self.regressor_x_scaler.transform(X)
            X = pd.DataFrame(X_train_norm, columns=X.columns)
        if self.regressor_scale_y:
            y = np.log(y + 1.01)
        self.regressor_gridsearch = GridSearchCV(estimator=RandomForestRegressor(),
                                                 param_grid=self.regressor_param_grid,
                                                 scoring='r2')
        self.regressor_gridsearch.fit(X=X, y=y)

    def get_regressor_params(self):
        return self.regressor_gridsearch.best_params_

    def get_classifier_params(self):
        return self.classifier_gridsearch.best_params_

    def _predict_classifier(self, X: np.ndarray) -> np.ndarray:
        if self.classifier_scale_x:
            scale_X = self.classifier_x_scaler.transform(X)
            X = pd.DataFrame(scale_X, columns=X.columns)

        pred_y = self.classifier_gridsearch.predict(X=X)
        return pred_y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        classifier_predict = self._predict_classifier(X)
        X = X.copy()
        # X["canceled"] = classifier_predict


        if self.regressor_scale_x:
            scale_X = self.regressor_x_scaler.transform(X)
            X = pd.DataFrame(scale_X, columns=X.columns)

        pred_y = self.regressor_gridsearch.predict(X=X)
        if self.regressor_scale_y:
            pred_y = np.exp(pred_y) - 1.01
        if self.regressor_scale_x:
            X = pd.DataFrame(self.regressor_x_scaler.inverse_transform(X), columns=X.columns)

        X["canceled"] = classifier_predict
        for index, row in X.iterrows():
            if row["canceled"] == 0:
                pred_y[index] = -1
        return pred_y


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        pass
