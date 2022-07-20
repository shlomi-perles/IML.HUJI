import numpy as np
import pandas as pd
from typing import Tuple, List
from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, Identity
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, FixedLR
from IMLearn.utils.utils import split_train_test
from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent.parent / "exrecise\ex7\plots"
FIGS_WIDTH = 650
FIGS_HEIGHT = 400
MARGINS_DICT = dict(l=5, r=0, t=24, b=0)


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class * n_classes, n_features)), np.zeros(samples_per_class * n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    split = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return tuple(map(lambda x: x.values, split))


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None, y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom, line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(title=rf"$\text{{Network Decision Boundaries {title}}}$",
                               xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray], lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[decision_surface(nn.predict, lims[0], lims[1], density=40, showscale=False),
                                     go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                                marker=dict(color=y, colorscale=custom,
                                                            line=dict(color="black", width=1)))
                                     ],
                               layout=go.Layout(title=rf"$\text{{{title} Iteration {i + 1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)


def get_callback(**kwargs):
    values = []
    grads = []
    out_weights = []

    def callback(**kwargs):
        values.append(kwargs["val"])
        grads.append(np.linalg.norm(kwargs["grad"]))
        out_weights.append(kwargs["weights"])

    return callback, values, grads, out_weights


def plot_and_show_nn(modules, out_dir, question_idx, hidden_size):
    save_end_name = f"_q{question_idx}_hidsiz{hidden_size}"
    callback, values, grads, weights = get_callback()
    nn = NeuralNetwork(
        modules=modules,
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(max_iter=5000, learning_rate=FixedLR(0.1), callback=callback))

    nn.fit(train_X, train_y)
    print(accuracy(test_y, nn.predict(test_X)))

    fig = plot_decision_boundary(nn, lims, train_X, train_y, title=out_dir.stem)
    fig['layout'].update(margin=MARGINS_DICT)
    fig.write_image(out_dir / f"decision_boundary{save_end_name}.svg")
    fig.show()

    save_name = OUT_DIR / f"animation{save_end_name}.gif" if len(weights) > 101 else None
    animate_decision_boundary(nn, weights[::100], lims, train_X, train_y, title="Simple Network",
                              save_name=save_name)

    plot_convergence(values, grads, hidden_size, modules, out_dir / f"convergence{save_end_name}.svg")


def plot_convergence(values, grads, hidden_size, modules, file_name):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=np.arange(1, len(values) + 1), y=values, name="Loss"))
    fig.add_trace(
        go.Scatter(x=np.arange(1, len(grads) + 1), y=grads, name="Norm"),
        secondary_y=True)
    fig.update_layout(width=FIGS_WIDTH, height=FIGS_HEIGHT,
                      title_text=rf"$\text{{Convergence Process, hidden size={hidden_size}, layers={len(modules)}}}$",
                      xaxis=dict(title=r"$\text{Iteration}$"))
    fig.update_yaxes(title_text=r"$\text{Loss}$", secondary_y=False)
    fig.update_yaxes(title_text=r"$\text{Norm}$", secondary_y=True)
    fig['layout'].update(margin=MARGINS_DICT)
    fig.write_image(file_name)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)

    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes, train_proportion=0.8)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    go.Figure(data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                               marker=dict(color=train_y, colorscale=custom, line=dict(color="black", width=1)))],
              layout=go.Layout(title=r"$\text{Train Data}$", xaxis=dict(title=r"$x_1$"), yaxis=dict(title=r"$x_2$"),
                               width=400, height=400)) \
        .write_image(OUT_DIR / "nonlinear_data.png")

    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#
    hidden_size = 16

    modules_lst = [[FullyConnectedLayer(input_dim=train_X.shape[1], output_dim=hidden_size, activation=ReLU(),
                                        include_intercept=True),
                    FullyConnectedLayer(input_dim=hidden_size, output_dim=hidden_size, activation=ReLU(),
                                        include_intercept=True),
                    FullyConnectedLayer(input_dim=hidden_size, output_dim=n_classes, activation=Identity(),
                                        include_intercept=True)],

                   [FullyConnectedLayer(input_dim=train_X.shape[1], output_dim=n_classes, activation=Identity(),
                                        include_intercept=True)]]

    # ---------------------------------------------------------------------------------------------#
    # Question 2: Fitting a network with no hidden layers                                          #
    # ---------------------------------------------------------------------------------------------#

    # ---------------------------------------------------------------------------------------------#
    # Question 3+4: Plotting network convergence process                                           #
    # ---------------------------------------------------------------------------------------------#

    for question_idx, modules in enumerate(modules_lst, start=1):
        for hid_s in {16, 6}:
            plot_and_show_nn(modules, OUT_DIR, question_idx, hid_s)
