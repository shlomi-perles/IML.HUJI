from IMLearn.utils import split_train_test
from IMLearn.tools import *
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent.parent / "datasets/house_prices.csv"
OUT_DIR = Path(__file__).parent.parent.parent / "exrecise\ex2\part1"
FIT_REPEAT = 10
PERCENTAGES = np.arange(0.1, 1.01, 0.01)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna(axis=0, how="any")
    df = df[(df.price > 0) & (df.sqft_living > 0) & (df.sqft_lot > 0)]
    response = df.price

    df.drop(labels=["id", "date", "price"], axis=1, inplace=True)
    df = make_dummies(df, ["zipcode"])

    return df, response


def make_dummies(df, columns):
    return pd.get_dummies(df, columns=columns)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for column_name, column_data in progressbar(X.iteritems()):
        plot_feature_evaluation(column_data, column_name, output_path, y)


def plot_feature_evaluation(column_data, column_name, output_path, y):
    cov = np.cov(column_data, y)[0][1]
    data_std = np.std(column_data)
    y_std = np.std(y)
    correlation = cov / (data_std * y_std)
    marker_size = 1 if correlation > 0.6 else 1.6
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    ax.scatter(column_data, y, s=marker_size, label="data")
    column_name = column_name.capitalize().replace(".", "_")
    y_name = y.name.capitalize()
    ax.set_title(
        f"{y_name} as Function of {column_name.capitalize()}\n"
        f"Pearson Correlation: {correlation:.3f}")
    # print(f"\nName :{column_name.capitalize()}, correlation = {round(correlation,3)}")
    ax.set_xlabel(column_name)
    ax.set_ylabel(y_name)
    ax.legend()
    fig.savefig(Path(output_path) / f"{y_name.lower()}_{column_name.lower()}.svg", format="svg", bbox_inches='tight')
    plt.close(fig)


def fit_model_by_percentage(train_X, train_y, test_X, test_y, percent, linear_regression, variance, average):
    """
    For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
      1) Sample p% of the overall training data
      2) Fit linear model (including intercept) over sampled set
      3) Test fitted model over test set
      4) Store average and variance of loss over test set
    """
    tmp_means = []
    for i in range(FIT_REPEAT):
        percent_train_X, percent_train_y, _, _ = split_train_test(train_X, train_y, percent)
        linear_regression.fit(percent_train_X, percent_train_y)
        tmp_means.append(linear_regression.loss(test_X, test_y))
    average.append(np.mean(tmp_means))
    variance.append(2 * np.std(tmp_means))


def plot_average_loss(variance, average):
    """
    plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    """
    fig, ax = plot_confidence_interval(PERCENTAGES, variance, average, label='mean')
    ax.set_title('Average Loss as Function of Part Of Training Size')
    ax.set_ylabel('Average Loss')
    ax.set_xlabel('Part Of Training Size')
    ax.legend()
    fig.savefig(OUT_DIR / "average_loss.svg", format="svg", bbox_inches='tight')


if __name__ == '__main__':
    np.random.seed(0)
    set_nicer_ploting()

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(str(DATA_PATH))

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, OUT_DIR)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression()
    variance = []
    average = []
    for percent in progressbar(PERCENTAGES):
        fit_model_by_percentage(train_X, train_y, test_X, test_y, percent, linear_regression, variance, average)
    plot_average_loss(np.array(variance), np.array(average))
