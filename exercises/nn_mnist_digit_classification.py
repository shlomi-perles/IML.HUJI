import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy, cross_entropy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax, Identity
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix
from nn_simulated_data import plot_convergence, OUT_DIR, FIGS_WIDTH, FIGS_HEIGHT, MARGINS_DICT
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import pickle

pio.templates.default = "simple_white"
CACHE_DIR = OUT_DIR.parent / "tmpFiles"
USE_CACHE = True
SAVE_CACHE = True


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


class Callback:
    def __init__(self, iterations=1, *args):
        self.idx_ = 0
        for key in args:
            setattr(self, key, np.zeros(iterations))

    def __call__(self, *args, **kwargs):
        for att, value in self.__dict__.items():
            if att.endswith("_"): continue
            if att == "times":
                value[self.idx_] = time.time()
                continue
            value[self.idx_] = kwargs[att]
        self.idx_ += 1


def plot_runtime_for_solver(train_X, train_y, solver, solver_name):
    nn_10 = NeuralNetwork(
        modules=[FullyConnectedLayer(input_dim=n_features, output_dim=hidden_size, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=hidden_size, output_dim=hidden_size, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=hidden_size, output_dim=n_classes, activation=Identity(),
                                     include_intercept=True)],
        loss_fn=CrossEntropyLoss(), solver=solver)

    pkl_file_name = CACHE_DIR / f"q10_{solver_name}_cache.pickle"
    nn_10 = fit_net(nn_10, train_X, train_y, pkl_file_name)

    times = np.array(nn_10.solver_.callback_.times) - nn_10.solver_.callback_.times[0]
    losses = np.array(nn_10.solver_.callback_.val)

    fig = go.Figure(data=[go.Scatter(x=times, y=losses)],
                    layout=go.Layout(width=FIGS_WIDTH, height=FIGS_HEIGHT,
                                     title=rf"$\text{{Runtime Of {solver_name}}}$",
                                     xaxis=dict(title=r"$\text{Runtime [s]}$"),
                                     yaxis=dict(title=r"$\text{Loss}$")))
    fig['layout'].update(margin=MARGINS_DICT)
    fig.write_image(OUT_DIR / f"q10_{solver_name}.svg")
    fig.show()
    return times, losses


def fit_net(nn, train_X, train_y, pkl_file_name):
    found_cache = False

    if USE_CACHE and pkl_file_name.exists():
        found_cache = True

        print(f"Loading cache for {pkl_file_name.stem}")
        with open(pkl_file_name, 'rb') as pkl_file:
            nn = pickle.load(pkl_file)
        print(f"Loading cache done.")
    else:
        nn.fit(train_X, train_y)

    if SAVE_CACHE and not found_cache:
        print(f"Save cache for {pkl_file_name.stem}")
        with open(pkl_file_name, 'wb') as pkl_file:
            pickle.dump(nn, pkl_file)
        print(f"Save done.")
    return nn


def q7(train_X, train_y, test_X, test_y, net_only=False):
    modules7 = [
        FullyConnectedLayer(input_dim=n_features, output_dim=hidden_size, activation=ReLU(), include_intercept=True),
        FullyConnectedLayer(input_dim=hidden_size, output_dim=hidden_size, activation=ReLU(), include_intercept=True),
        FullyConnectedLayer(input_dim=hidden_size, output_dim=n_classes, activation=Identity(), include_intercept=True)]
    nn = NeuralNetwork(modules=modules7, loss_fn=CrossEntropyLoss(),
                       solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                                        callback=Callback(10000, "val", "grad", "weights")))

    pkl_file_name = CACHE_DIR / "q7_cache.pickle"
    nn = fit_net(nn, train_X, train_y, pkl_file_name)
    if net_only: return nn
    pred = nn.predict(test_X)
    print(accuracy(test_y, pred))

    save_end_name = f"_q7_hidsiz{hidden_size}"
    plot_convergence(nn.solver_.callback_.val, nn.solver_.callback_.grad, hidden_size, modules7,
                     OUT_DIR / f"convergence{save_end_name}.svg")
    print(confusion_matrix(pred, test_y))
    return nn


def q8(train_X, train_y, test_X, test_y):
    modules8 = [FullyConnectedLayer(input_dim=n_features, output_dim=n_classes, activation=Identity(),
                                    include_intercept=True)]
    nn8 = NeuralNetwork(modules=modules8, loss_fn=CrossEntropyLoss(),
                        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256))

    pkl_file_name = CACHE_DIR / "q8_cache.pickle"
    nn8 = fit_net(nn8, train_X, train_y, pkl_file_name)

    pred = nn8.predict(test_X)
    print(accuracy(test_y, pred))


def q9(nn, test_X, test_y):
    test_X_7 = test_X[test_y == 7]

    pred = nn.compute_prediction(test_X_7).max(axis=1).argsort()
    margins = MARGINS_DICT.copy()
    margins["t"] = 38
    margins["l"] = 0
    fig1 = plot_images_grid(test_X_7[pred[-64:]], title="Most Confident")
    fig1['layout'].update(width=FIGS_HEIGHT, height=FIGS_HEIGHT, margin=margins)
    fig1.write_image(OUT_DIR / f"q9_most.svg")
    fig1.show()

    fig2 = plot_images_grid(test_X_7[pred[:64]], title="Least Confident")
    fig2['layout'].update(width=FIGS_HEIGHT, height=FIGS_HEIGHT, margin=margins)
    fig2.write_image(OUT_DIR / f"q9_least.svg")
    fig2.show()


def q10(train_X, train_y):
    train_X_q10 = train_X[:2500]
    train_y_q10 = train_y[:2500]
    times_gd, losses_gd = plot_runtime_for_solver(train_X_q10, train_y_q10,
                                                  GradientDescent(max_iter=10000, learning_rate=FixedLR(0.1),
                                                                  tol=1e-10,
                                                                  callback=Callback(10000, "times", "val")), "GD")
    times_sgd, losses_sgd = plot_runtime_for_solver(train_X_q10, train_y_q10,
                                                    StochasticGradientDescent(max_iter=10000,
                                                                              learning_rate=FixedLR(0.1),
                                                                              tol=1e-10, batch_size=64,
                                                                              callback=Callback(10000, "times", "val"))
                                                    , "SGD")
    fig = go.Figure(
        data=[go.Scatter(x=times_gd, y=losses_gd, name='GD'), go.Scatter(x=times_sgd, y=losses_sgd, name='SGD')],
        layout=go.Layout(width=FIGS_WIDTH, height=FIGS_HEIGHT, title=r"$\text{Runtime Differences Between SGD And GD}$",
                         xaxis=dict(title=r"$\text{Runtime [s]}$"),
                         yaxis=dict(title=r"$\text{Loss}$")))
    fig['layout'].update(margin=MARGINS_DICT)
    fig.write_image(OUT_DIR / f"q10_runtime_diff.svg")
    fig.show()


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    hidden_size = 64

    nn = q7(train_X, train_y, test_X, test_y)

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    q8(train_X, train_y, test_X, test_y)

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#

    q9(nn, test_X, test_y)

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    q10(train_X, train_y)
