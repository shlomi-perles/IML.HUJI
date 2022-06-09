from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from IMLearn.tools import *
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

SAVE_FIGS = True
PLOT_FIGS = False
OUT_DIR = Path(__file__).parent.parent.parent / "exrecise\ex5\plots"
X_RANGE = (-1.2, 2)
LAMBDA_RANGE = (1e-2, 3)
MAX_DEGREE = 10


def f(X, gaussian_noise=0):
    return (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2) + gaussian_noise


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    X = np.linspace(*X_RANGE, n_samples)
    y = f(X, np.random.randn(n_samples) * noise)
    train_X, train_y, test_X, test_y = split_train_test(np.array(X), np.array(y), 2 / 3)

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    q1(X, test_X, test_y, train_X, train_y, noise)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_scores = q2(train_X, train_y, noise)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    q3(test_X, test_y, train_X, train_y, validation_scores)


def q3(test_X, test_y, train_X, train_y, validation_scores):
    k_star = np.argmin(validation_scores)
    poly_fit = PolynomialFitting(k_star)
    poly_fit.fit(train_X, train_y)
    print(rf"k ^ {{\star}} &= {k_star} \\test\ error &= {poly_fit.loss(test_X, test_y)}")


def q2(train_X, train_y, noise):
    training_scores = []
    validation_scores = []
    for degree in range(MAX_DEGREE + 1):
        train_score, validation_score = cross_validate(PolynomialFitting(degree), train_X, train_y, mean_square_error,
                                                       cv=5)
        training_scores.append(train_score)
        validation_scores.append(validation_score)
    fig, ax = plt.subplots()
    ax.scatter(list(range(MAX_DEGREE + 1)), training_scores, label="training set")
    ax.scatter(list(range(MAX_DEGREE + 1)), validation_scores, label="validation set")
    ax.set_title("Average Error as function of the polynomial degree")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Average Error")
    ax.legend()
    if SAVE_FIGS:
        fig.savefig(OUT_DIR / f"q2_noise_{noise}.svg", format="svg", bbox_inches='tight')
    if PLOT_FIGS:
        plt.show()
        plt.close(fig)
    return validation_scores


def q1(X, train_X, train_y, test_X, test_y, noise):
    fig, ax = plt.subplots()
    ax.scatter(train_X, train_y, label="training set")
    ax.scatter(test_X, test_y, label="validation set")
    ax.scatter(X, f(X), color="black", label="noiseless model")
    ax.set_title(f"Scatter plot of the true (noiseless) model and the two sets")
    ax.set_xlabel("X")
    ax.set_ylabel("f(X)")
    ax.legend()
    if SAVE_FIGS:
        fig.savefig(OUT_DIR / f"q1_noise_{noise}.svg", format="svg", bbox_inches='tight')
    if PLOT_FIGS:
        plt.show()
        plt.close(fig)
    return test_X, test_y, train_X, train_y


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    test_X, test_y, train_X, train_y = q6(n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range, lasso_validation_scores, ridge_validation_scores = q7(n_evaluations, train_X, train_y)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    q8(lambda_range, lasso_validation_scores, ridge_validation_scores, test_X, test_y, train_X, train_y)


def q8(lambda_range, lasso_validation_scores, ridge_validation_scores, test_X, test_y, train_X, train_y):
    best_ridge_idx, best_lasso_idx = np.argmin(ridge_validation_scores), np.argmin(lasso_validation_scores)
    best_ridge, best_lasso = RidgeRegression(lam=lambda_range[best_ridge_idx]), Lasso(
        alpha=lambda_range[best_lasso_idx])
    best_ridge.fit(train_X, train_y)
    best_lasso.fit(train_X, train_y)
    lasso_pred = best_lasso.predict(test_X)
    best_error_ridge, best_error_lasso = best_ridge.loss(test_X, test_y), mean_square_error(lasso_pred, test_y)
    linear = LinearRegression(include_intercept=True)
    linear.fit(train_X, train_y)
    best_error_linear = linear.loss(test_X, test_y)
    print(f"Ridge	{lambda_range[best_ridge_idx]}	{best_error_ridge}")
    print(f"Lasso	{lambda_range[best_lasso_idx]}	{best_error_lasso}")
    print(f"Linear		{best_error_linear}")


def q7(n_evaluations, train_X, train_y):
    lambda_range = np.linspace(*LAMBDA_RANGE, num=n_evaluations)
    ridge_train_scores = []
    ridge_validation_scores = []
    lasso_train_scores = []
    lasso_validation_scores = []
    for lam in lambda_range:
        ridge = RidgeRegression(lam=lam)
        ridge_train_score, ridge_validation_score = cross_validate(ridge, train_X, train_y,
                                                                   mean_square_error, cv=5)

        ridge_train_scores.append(ridge_train_score)
        ridge_validation_scores.append(ridge_validation_score)

        lasso = Lasso(alpha=lam)
        lasso_train_score, lasso_validation_score = cross_validate(lasso, train_X, train_y,
                                                                   mean_square_error, cv=5)
        lasso_train_scores.append(lasso_train_score)
        lasso_validation_scores.append(lasso_validation_score)

    fig, ax = plt.subplots()
    ax.scatter(lambda_range, ridge_train_scores, label="training set")
    ax.scatter(lambda_range, ridge_validation_scores, label="validation set")
    ax.set_title(fr"Ridge Average Error as function of $\lambda$")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("Average Error")
    ax.legend()
    if SAVE_FIGS:
        fig.savefig(OUT_DIR / f"q7_ridge.svg", format="svg", bbox_inches='tight')
    if PLOT_FIGS:
        plt.show()
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(lambda_range, lasso_train_scores, label="training set")
    ax.scatter(lambda_range, lasso_validation_scores, label="validation set")
    ax.set_title(fr"Lasso Average Error as function of $\lambda$")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("Average Error")
    ax.legend()
    if SAVE_FIGS:
        fig.savefig(OUT_DIR / f"q7_lasso.svg", format="svg", bbox_inches='tight')
    if PLOT_FIGS:
        plt.show()
        plt.close(fig)

    return lambda_range, lasso_validation_scores, ridge_validation_scores


def q6(n_samples):
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X = X[:n_samples]
    train_y = y[:n_samples]
    test_X = X[n_samples:]
    test_y = y[n_samples:]
    return test_X, test_y, train_X, train_y


if __name__ == '__main__':
    np.random.seed(0)
    set_nicer_ploting()
    select_polynomial_degree()
    # Question 4
    select_polynomial_degree(noise=0)
    # Question 5
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
