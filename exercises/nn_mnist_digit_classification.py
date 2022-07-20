import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy, cross_entropy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax, Identity
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix
from nn_simulated_data import get_callback, plot_convergence, OUT_DIR, Path
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


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


def q8():
    global callback, values, grads, out_weights, pred
    modules8 = [FullyConnectedLayer(input_dim=n_features, output_dim=n_classes, activation=Identity(),
                                    include_intercept=True)]
    callback, values, grads, out_weights = get_callback()
    nn8 = NeuralNetwork(modules=modules8, loss_fn=CrossEntropyLoss(),
                        solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256))
    nn8.fit(train_X, train_y)
    pred = nn.predict(test_X)
    print(accuracy(test_y, pred))


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    y_pred = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float)
    y_true = np.array([0, 0, 0, 1], dtype=np.float)

    # y_pred = np.array([[0.25, 0.25, 0.25, 0.25],
    #                    [0.01, 0.01, 0.01, 0.96]], dtype=np.float)
    # y_true = np.array([[0, 0, 0, 1],
    #                    [0, 0, 0, 1]], dtype=np.float)
    ans = 0.71355817782  # Correct answer
    x = cross_entropy(y_true, y_pred)
    from sklearn.metrics import log_loss

    c = log_loss(y_true, y_pred)
    a = 1
    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    hidden_size = 64
    modules7 = [
        FullyConnectedLayer(input_dim=n_features, output_dim=hidden_size, activation=ReLU(), include_intercept=True),
        FullyConnectedLayer(input_dim=hidden_size, output_dim=hidden_size, activation=ReLU(), include_intercept=True),
        FullyConnectedLayer(input_dim=hidden_size, output_dim=n_classes, activation=Identity(), include_intercept=True)]

    callback, values, grads, out_weights = get_callback()

    nn = NeuralNetwork(modules=modules7, loss_fn=CrossEntropyLoss(),
                       solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                                        callback=callback))

    nn.fit(train_X, train_y)
    pred = nn.predict(test_X)
    print(accuracy(test_y, pred))

    save_end_name = f"_q{7}_hidsiz{hidden_size}"
    # Plotting convergence process
    plot_convergence(values, grads, hidden_size, modules7, OUT_DIR / f"convergence{save_end_name}.svg")

    print(confusion_matrix(pred, test_y))

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    q8()

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#

    test_X = test_X[test_y == 7]
    test_y = test_y[test_y == 7]

    # get 64 most confident samples
    confidences = nn.compute_prediction(test_X)
    seven_sorted_pics = np.argsort(np.max(pred_7, axis=1))

    best = plot_images_grid(test_X_7[seven_sorted_pics[:64], :])
    worst = plot_images_grid(test_X_7[seven_sorted_pics[-64:], :])

    # plot 64 most confident samples
    im = plot_images_grid(test_X[np.argsort(confidences)[-64:]].reshape(64, 784), title="Most confident")
    im.show()
    im = plot_images_grid(test_X[np.argsort(confidences)[:64]].reshape(64, 784), title="Least confident")
    im.show()

    inds7 = np.where(test_y == 7)[0]
    samples_of_7 = test_X[inds7]
    nn.compute_prediction(samples_of_7)
    predictions = nn.post_activations[-1]
    predicted_probs_of_7 = predictions[:, 7]
    sorted_inds = np.argsort(predicted_probs_of_7)
    most_confident = sorted_inds[-64:]
    least_confident = sorted_inds[:64]
    fig1 = plot_images_grid(samples_of_7[most_confident], title="Most Confident")
    fig1.show()
    fig2 = plot_images_grid(samples_of_7[least_confident], title="Least Confident")
    fig2.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    # raise NotImplementedError()
