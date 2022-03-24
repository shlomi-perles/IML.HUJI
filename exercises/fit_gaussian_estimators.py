from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt

pio.templates.default = "simple_white"
SAMPLES = 1000
QUESTION_ONE_MEAN = 10
QUESTION_ONE_VAR = 1
QUESTION_ONE_SAMPLES_SKIP = 10

QUESTION_TWO_RESOLUTION = 200
QUESTION_TWO_GRID_SIZE = 10


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(QUESTION_ONE_MEAN, QUESTION_ONE_VAR, size=SAMPLES)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(samples)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    x = np.arange(QUESTION_ONE_MEAN, SAMPLES + 1, QUESTION_ONE_SAMPLES_SKIP)
    estimate_mean_dis = np.vectorize(lambda last_index: np.abs(np.mean(samples[:last_index]) - QUESTION_ONE_MEAN))
    fig = go.Figure(
        [go.Scatter(x=x, y=estimate_mean_dis(x), mode='markers', name=r'$\left|\hat{\mu}(m)-10\right|$',
                    showlegend=True)], layout=go.Layout(
            title={
                "text": r"$\text{Distance Between The Estimated-And True Value Of The Expectations}\\"
                        r"\text{As Function Of Number Of Samples}$",
                'y': 0.84, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title=r"$\text{Number of samples} [m]$", yaxis_title=r"$\left|\hat{\mu}(m)-10\right|$", height=400))
    fig.show()
    # fig.write_image("estimate_distance.svg")

    # Question 3 - Plotting Empirical PDF of fitted model
    fig = go.Figure(
        [go.Scatter(x=samples, y=univariate_gaussian.pdf(samples), mode='markers',
                    showlegend=False, marker=dict(size=2))], layout=go.Layout(
            title={
                "text": r"$\text{Probability Density As Function Of Samples Values}$",
                'y': 0.84, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
            xaxis_title=r"$\text{Sample value}$", yaxis_title=r"$\text{Probability density}$", height=400))
    fig.show()
    # fig.write_image("pdf_q1.svg")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, SAMPLES)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-QUESTION_TWO_GRID_SIZE, QUESTION_TWO_GRID_SIZE, QUESTION_TWO_RESOLUTION)
    grid_tuples = np.transpose(np.array([np.repeat(f1, len(f1)), np.tile(f1, len(f1))]))
    calc_log_likelihood = lambda x1, x3: multivariate_gaussian.log_likelihood(np.array([x1, 0, x3, 0]), sigma, samples)
    Z = np.vectorize(calc_log_likelihood)(grid_tuples[:, 0], grid_tuples[:, 1]).reshape(QUESTION_TWO_RESOLUTION,
                                                                                        QUESTION_TWO_RESOLUTION)
    fig, ax = plt.subplots()
    heat_map = ax.pcolormesh(f1, f1, Z)
    fig.colorbar(heat_map, format='%.e')
    ax.set_title("log-likelihood for " + r"$\mu=\left[f_{1},0,f_{3},0\right]{}^{T}$")
    ax.set_xlabel("$f_{3}$")
    ax.set_ylabel("$f_{1}$")
    plt.show()


    # Question 6 - Maximum likelihood
    max_coordinates = np.where(Z == np.amax(Z))
    print(f"({round(f1[max_coordinates[0]][0], 3)}, {round(f1[max_coordinates[1]][0], 3)})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
