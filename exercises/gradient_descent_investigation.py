import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.io as pio
from IMLearn.tools import *

pio.templates.default = "simple_white"
OUT_DIR = Path(__file__).parent.parent.parent / "exrecise\ex6\plots"
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = list()
    weights = list()

    def callback(**kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    callback, values, weights = get_gd_state_recorder_callback()
    for name, modul in zip(["L1", "L2"], [L1, L2]):
        for eta in etas:
            reg = modul(init)
            gradient = GradientDescent(FixedLR(eta), callback=callback)
            gradient.fit(reg, None, None)
            path = np.stack(weights, axis=0)

            color_fig = plot_descent_path(modul, path, f"{name}, eta={eta}")
            color_fig.write_image(OUT_DIR / f"check_{name}_{eta}.svg")

            fig, ax = plt.subplots()
            ax.scatter(list(range(len(values))), values)
            ax.set_title(f"$\eta={eta}$")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("$values$")
            fig.savefig(OUT_DIR / f"check_{name}_{eta}_graph.svg", bbox_inches='tight')
            plt.close(fig)
            values.clear()
            weights.clear()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig, ax = plt.subplots()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("values")

    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        gradient = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gradient.fit(L1(init), None, None)
        plt.scatter(list(range(len(values))), values, label=f"$\gamma={gamma}$")
    # Plot algorithm's convergence for the different values of gamma
    fig.legend()
    fig.savefig(OUT_DIR / f"exp_gamma_vals.svg", bbox_inches='tight')
    plt.clf()

    # Plot descent path for gamma=0.95
    reg = L1(init)
    lt = ExponentialLR(eta, 0.95)
    callback, values, weights = get_gd_state_recorder_callback()
    gradient = GradientDescent(lt, callback=callback)
    gradient.fit(reg, None, None)
    fig = plot_descent_path(L1, np.array(weights), f"{type(reg)}, {eta}")
    fig.write_image(OUT_DIR / f"exp_path.svg")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    reg_modul = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000))
    y_pred = reg_modul.fit(X_train, y_train).predict_proba(X_train)

    false_pos_lst = list()
    true_pos_lst = list()
    for alpha in range(0, 101):
        y_alpha = (y_pred > alpha * 0.01).astype(int)
        true_neg = np.sum(np.logical_and(y_alpha == y_train, np.logical_not(y_alpha)))
        true_pos = np.sum(np.logical_and(y_alpha == y_train, y_alpha))
        false_pos = np.sum(np.logical_and(y_alpha != y_train, y_alpha))
        false_neg = np.size(y_alpha) - true_pos - true_neg - false_pos

        div = 1 if false_pos == 0 else false_pos + true_neg
        add_pos = 0 if true_pos == 0 else true_pos / (false_neg + true_pos)
        false_pos_lst.append(false_pos / div)
        true_pos_lst.append(add_pos)

    # Plotting convergence rate of logistic regression over SA heart disease data
    fig, ax = plt.subplots()
    ax.scatter(false_pos_lst, true_pos_lst)
    ax.set_title("ROC curve")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    fig.savefig(OUT_DIR / f"rock_curve.svg", bbox_inches='tight')
    plt.close(fig)

    best_alpha = 0.01 * np.argmax(np.array(true_pos_lst) - np.array(false_pos_lst))
    reg_modul.alpha_ = best_alpha
    best_alpha_loss = reg_modul.loss(X_test, y_test)
    print(rf"Best \alpha={best_alpha}")
    print(rf"Best \alpha_{{loss}}={best_alpha_loss}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambda_lst = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in ["l1", "l2"]:
        scores_lst = list()
        for lam in lambda_lst:
            log = LogisticRegression(penalty=penalty, alpha=0.5, lam=lam)
            train_score, validation_score = cross_validate(log, X_train, y_train, misclassification_error)
            scores_lst.append(validation_score)

        min_lambda = lambda_lst[scores_lst.index(min(scores_lst))]
        log = LogisticRegression(penalty=penalty, alpha=0.5, lam=min_lambda)
        log.fit(X_train, y_train)
        err = log.loss(X_test, y_test)
        print(f"penalty={penalty}, Best \lambda={min_lambda}")
        print(f"penalty={penalty}, Best error={err}")


if __name__ == '__main__':
    np.random.seed(0)
    set_nicer_ploting()
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
