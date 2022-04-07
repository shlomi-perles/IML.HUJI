import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from IMLearn.tools import *

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from pathlib import Path
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"
DATA_PATH = Path(__file__).parent.parent / "datasets/City_Temperature.csv"
OUT_DIR = Path(__file__).parent.parent.parent / "exrecise\ex2\part2"
POLYNOM_DEGREES = list(range(1, 11))


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename)
    df = df.dropna(axis=0, how="any")
    dates = pd.to_datetime(df.Date)
    df = df[(df.Country != "") & (df.City != "") &
            (df.Year > 0) &
            (df.Month <= 12) & (0 < df.Month) &
            (df.Day <= 31) & (0 < df.Day) &
            (-70 < df.Temp) &
            (dates.dt.year == df.Year) & (dates.dt.month == df.Month) & (dates.dt.day == df.Day)]
    df["DayOfYear"] = pd.to_datetime(df.Date).dt.day_of_year
    return df


def question_2(df):
    df = df[df.Country == "Israel"]
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    for year, frame in df.groupby(['Year']):
        ax.scatter(frame.DayOfYear, frame.Temp, s=1.2, label=year)

    ax.set_title("Temperature as Function of Day of Year")
    ax.set_ylabel("Temperature")
    ax.set_xlabel("Day of Year")
    ax.legend()
    fig.savefig(OUT_DIR / f"q1_temp_vs_day.svg", format="svg", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    month_df = df[["Temp", "Month"]]
    month_std = month_df.groupby(["Month"]).agg("std")
    ax.bar([i for i in range(1, len(month_std) + 1)], month_std.Temp)
    ax.set_title("Std as Function of Month")
    ax.set_ylabel("Std")
    ax.set_xlabel("Month")
    fig.savefig(OUT_DIR / f"q1_STD_vs_month.svg", format="svg", bbox_inches='tight')
    plt.close(fig)


def question_3(df):
    data = df.drop(["Year", "Day", "DayOfYear"], axis=1).groupby(["Month", "Country"]).agg(Mean=('Temp', 'mean'),
                                                                                           Std=('Temp', 'std'))
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid()
    for country, frame in data.groupby(["Country"]):
        ax.errorbar(frame.index.levels[0], frame.Mean, yerr=frame.Std, label=country)
    ax.set_title("Temperature as function of Month")
    ax.set_ylabel("Average Temperature")
    ax.set_xlabel("Month")
    ax.legend()
    fig.savefig(OUT_DIR / f"q3_Temp_vs_Month.svg", format="svg", bbox_inches='tight')
    plt.close(fig)


def question_4(df):
    df = df[df.Country == "Israel"]
    train_X, train_y, test_X, test_y = split_train_test(df.DayOfYear, df.Temp, train_proportion=0.75)
    loss_lst = []
    for degree in POLYNOM_DEGREES:
        loss_lst.append(PolynomialFitting(degree).fit(train_X, train_y).loss(test_X, test_y))
        print(f"Degree = {degree},Loss = {round(loss_lst[-1],2)}")

    fig, ax = plt.subplots()
    ax.bar(POLYNOM_DEGREES, loss_lst)
    ax.set_title("Loss as Function of Polynom Degree")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Polynom Degree")
    fig.savefig(OUT_DIR / f"q4_degree_vs_loss.svg", format="svg", bbox_inches='tight')
    plt.close(fig)
    return loss_lst.index(min(loss_lst)) + 1


def question_5(df, degree):
    print(f"Best degree: {degree}")
    israel_df = df[df.Country == "Israel"]
    poly_fit = PolynomialFitting(degree).fit(israel_df.DayOfYear, israel_df.Temp)
    loss_dict = {}
    for country, frame in df.groupby(["Country"]):
        if country != "Israel":
            loss_dict[country] = poly_fit.loss(frame.DayOfYear, frame.Temp)

    fig, ax = plt.subplots()
    ax.bar(*zip(*loss_dict.items()))
    ax.set_title("Loss as Function of Country")
    ax.set_ylabel("Loss")
    ax.set_xlabel("County")
    fig.savefig(OUT_DIR / f"q5_loss_vs_county.svg", format="svg", bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    np.random.seed(0)
    set_nicer_ploting()
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(str(DATA_PATH))

    # Question 2 - Exploring data for specific country
    question_2(df)

    # Question 3 - Exploring differences between countries
    question_3(df)

    # Question 4 - Fitting model for different values of `k`
    degree = question_4(df)

    # Question 5 - Evaluating fitted model on different countries
    question_5(df, degree)
