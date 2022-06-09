from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    groups = np.random.permutation(np.remainder(np.arange(y.size), cv))
    train_score = []
    validation_score = []

    for i in range(cv):
        msk = groups == i
        test_X, test_y = X[msk], y[msk]
        train_X, train_y = X[~msk], y[~msk]

        train_estimator = estimator.fit(train_X, train_y)
        validation_score.append(scoring(train_estimator.predict(test_X), test_y))
        train_score.append(scoring(train_estimator.predict(train_X), train_y))

    return sum(train_score) / cv, sum(validation_score) / cv
