from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy
from pathlib import Path
from IMLearn.tools import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import re

DATA_PATH = Path(__file__).parent.parent / "datasets"
OUT_DIR = Path(__file__).parent.parent.parent / "exrecise/ex3/"
MESHGRID_STEPS = 0.01
GAUS_DATASET = ["gaussian1.npy", "gaussian2.npy"]
PERCEP_DATASET = ["linearly_separable.npy", "linearly_inseparable.npy"]


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for dataset in PERCEP_DATASET:
        # Load dataset
        data_path = DATA_PATH / dataset
        X, y = load_dataset(str(data_path))

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(perceptron_est, *args):
            losses.append(perceptron_est.loss(X, y))

        Perceptron(callback=callback, include_intercept=True).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        plt.clf()
        re_dataset = re.split(r"_|\.", dataset)
        dataset_name = (re_dataset[0] + " " + re_dataset[1]).title()
        plt.plot(list(range(1, len(losses) + 1)), losses)
        plt.xlabel("Fitting Iteration")
        plt.ylabel("Loss")
        plt.title(f"Loss of as function of Fitting Iteration - {dataset_name}")
        plt.show()
        plt.clf()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return plt.plot(mu[0] + xs, mu[1] + ys, marker="_", color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for dataset in GAUS_DATASET:
        # Load dataset
        data_path = DATA_PATH / dataset
        X, y = load_dataset(str(data_path))

        # Fit models and predict over training set
        lda, gnb = LDA(), GaussianNaiveBayes()
        estimators = {lda, gnb}
        lda.fit(X, y)
        gnb.fit(X, y)
        colors = ("orange", "blue", "red")

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        for estimator in estimators:
            X1, X2 = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, MESHGRID_STEPS),
                                 np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, MESHGRID_STEPS))
            # Add traces for data-points setting symbols and colors
            for cls, j in enumerate(np.unique(y)):
                plt.scatter(X[y == j, 0], X[y == j, 1],
                            color=ListedColormap(colors)(cls), label=j)

            # Add `X` dots specifying fitted Gaussians' means

            plt.contourf(X1, X2, estimator.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                         alpha=0.4, cmap=ListedColormap(colors))
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            plt.legend(frameon=True)
            plt.axis('off')

            y_pred = estimator.predict(X)

            # Add ellipses depicting the covariances of the fitted Gaussians
            for cls in range(len(estimator.classes_)):
                plt.scatter(estimator.mu_[cls][0], estimator.mu_[cls][1], marker="x", color="black")

                if estimator == lda:
                    plt.title(f"LDA predict, accuracy = ${accuracy(y, y_pred)}$")
                    get_ellipse(estimator.mu_[cls], estimator.cov_)
                else:
                    plt.title(f"Gaussian predict, accuracy = ${accuracy(y, y_pred)}$")
                    get_ellipse(estimator.mu_[cls], np.array([[gnb.vars_[cls][0], 0], [0, gnb.vars_[cls][1]]]))
            plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    set_nicer_ploting()
    run_perceptron()
    compare_gaussian_classifiers()
