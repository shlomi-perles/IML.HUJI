from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.tools import *
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent.parent / "exrecise\ex4\plots"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plot_decision_boundary(grid, color, adaboost, lims, i, D=None):
    return [decision_surface(lambda X: adaboost.partial_predict(X, i), lims[0], lims[1], showscale=False),
            go.Scatter(x=grid[:, 0], y=grid[:, 1], mode="markers",
                       marker=dict(color=color, symbol=class_symbols[color.astype(np.int32)], size=D,
                                   colorscale=custom))
            ]


def q1(train_X, train_y, test_X, test_y, adaboost, noise, n_learners):
    test_los = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
    train_los = [adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
    marker_size = 10
    fig, ax = plt.subplots()
    ax.scatter(list(range(1, n_learners + 1)), test_los, label="test", s=marker_size)
    ax.scatter(list(range(1, n_learners + 1)), train_los, label="train", s=marker_size)
    ax.set_title(f"Error as function of the Number Of Fitted Learners")
    ax.set_xlabel("Number Of Fitted Learners")
    ax.set_ylabel("Error")
    ax.legend()
    fig.savefig(OUT_DIR / f"q1_noise_{noise}.svg", format="svg", bbox_inches='tight')
    plt.close(fig)


def q2(T, adaboost, lims, noise, test_X, test_y):
    q2_fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, subplot_titles=T,
                           vertical_spacing=0.07, horizontal_spacing=0.02)
    position = [(x, y) for x in range(1, 3) for y in range(1, 3)]
    for pos, i in enumerate(T):
        for trace in plot_decision_boundary(test_X, test_y, adaboost, lims, i, 3):
            q2_fig.add_trace(trace, row=position[pos][0], col=position[pos][1])
    q2_fig.update_layout(title_text="Decision Boundary", showlegend=False)
    q2_fig.write_image(str(OUT_DIR / f"q2_noise_{noise}.svg"))


def q3(test_X, test_y, adaboost, noise, lims, n_learners):
    min_err = 1
    min_iter = 0
    for i in range(1, n_learners + 1):
        err = adaboost.partial_loss(test_X, test_y, i)
        if err < min_err:
            min_err = err
            min_iter = i
    fig = go.Figure(layout=go.Layout(title=rf"Decision Boundary | Iteration={min_iter}, Accuracy={1 - min_err}"))
    for trace in plot_decision_boundary(test_X, test_y, adaboost, lims, min_iter):
        fig.add_trace(trace)
    fig.write_image(str(OUT_DIR / f"q3_noise_{noise}.svg"))


def q4(adaboost, lims, noise, train_X, train_y):
    factor = 6 if noise else 18
    fig = go.Figure(layout=go.Layout(title=rf"Decision Boundary"))
    for trace in plot_decision_boundary(train_X, train_y, adaboost, lims, 250,
                                        adaboost.D_ / np.max(adaboost.D_) * factor):
        fig.add_trace(trace)
    fig.update_layout(xaxis_range=[-1, 1], yaxis_range=[-1, 1])
    fig.write_image(str(OUT_DIR / f"q4_noise_{noise}.svg"))


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    q1(train_X, train_y, test_X, test_y, adaboost, noise, n_learners)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    q2(T, adaboost, lims, noise, test_X, test_y)

    # Question 3: Decision surface of best performing ensemble
    q3(test_X, test_y, adaboost, noise, lims, n_learners)

    # Question 4: Decision surface with weighted samples
    q4(adaboost, lims, noise, train_X, train_y)


if __name__ == '__main__':
    np.random.seed(0)
    set_nicer_ploting()
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
