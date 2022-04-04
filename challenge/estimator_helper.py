from sklearn.ensemble import RandomForestRegressor
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt  # some plotting!
from scipy import stats
from subprocess import check_output
from scipy.stats import norm
import plotly.figure_factory as ff


def check_data_norm(target_vector, apply_log=False, both=True):
    log_data = np.array([0, 0.001])
    plot_name = "plot"
    if both:
        log_data = np.log(target_vector)
    elif apply_log:
        target_vector = np.log(target_vector)
        plot_name = "log"

    m = target_vector.mean()
    s = target_vector.std()
    gaussian_data = np.random.normal(m, s, 10000)

    if both:
        m_log = target_vector.mean()
        s_log = target_vector.std()
        gaussian_data_log = np.random.normal(m_log, s_log, 10000)

    if both:
        fig = ff.create_distplot(
            [target_vector, log_data, gaussian_data, gaussian_data_log],
            group_labels=["plot", "log", "gaussian", "gaussian log"],
            curve_type="kde")
    else:
        fig = ff.create_distplot(
            [target_vector, gaussian_data],
            group_labels=[plot_name, "gaussian"],
            curve_type="kde")
    fig.update_layout(showlegend=True)
    fig.show()

    plt_fig = plt.figure()
    plt_fig.set_size_inches(10, 5)

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    res = stats.probplot(target_vector, plot=ax1)
    ax1.set_title("Probability Plot", fontsize=14)
    ax1.set_ylabel("Sample Quantiles", fontsize=12)
    ax1.set_xlabel("Theoretical Quantiles", fontsize=12)

    if both:
        res = stats.probplot(log_data, plot=ax2)
        ax2.set_title("Probability Plot of log", fontsize=14)
        ax2.set_ylabel("Sample Quantiles", fontsize=12)
        ax2.set_xlabel("Theoretical Quantiles", fontsize=12)
        ax2.legend()

    plt.show()


def plot_feature_importance(X, y):
    mir_result = mutual_info_regression(X, y)  # mutual information regression feature ordering
    feature_scores = []
    for i in np.arange(len(X.columns)):
        feature_scores.append([X.columns[i], mir_result[i]])
    sorted_scores = sorted(np.array(feature_scores), key=lambda s: float(s[1]), reverse=True)
    print(np.array(sorted_scores))
    # for i, col in enumerate(sorted_scores):
    #     print(f"{i}: ", X.columns[int(col[0])])
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    ind = np.arange(len(X.columns))
    plt.bar([col[0] for col in sorted_scores], [float(i) for i in np.array(sorted_scores)[:, 1]])
    ax.axes.set_xticks(X.columns)
    plt.title('Feature Importances (Mutual Information Regression)')
    plt.ylabel('Importance')
    plt.show()

