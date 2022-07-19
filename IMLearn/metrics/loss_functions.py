import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.square(np.subtract(y_true, y_pred)).mean()


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    miss_loss = (y_true != y_pred).sum()
    return miss_loss / len(y_true) if normalize else miss_loss


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    return (y_true == y_pred).mean()


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    # from sklearn.metrics import log_loss
    # return log_loss(y_true, y_pred)

    epsilon = 1e-5
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # if y_true.ndim == 1:
    #     y_true = y_true[:, np.newaxis]
    # if y_true.shape[1] == 1:
    #     y_true = np.append(1 - y_true, y_true, axis=1)
    # if y_pred.ndim == 1:
    #     y_pred = y_pred[:, np.newaxis]
    # if y_pred.shape[1] == 1:
    #     y_pred = np.append(1 - y_pred, y_pred, axis=1)

    dup_y_true = np.zeros_like(y_pred)
    dup_y_true[np.arange(len(y_pred)), y_true] = 1

    return -(dup_y_true * np.log(y_pred)).sum() / y_pred.shape[0]  # TODO: axis=1?


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
