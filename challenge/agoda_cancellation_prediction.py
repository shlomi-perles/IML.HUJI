from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import pickle
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
# from tqdm import tqdm
from sklearn.model_selection import train_test_split

BEGIN_CHECK_DATE = datetime.date(2018, 12, 7)
END_CHECK_DATE = datetime.date(2018, 12, 13)


def load_data1(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    __add_time_between_booking_to_cancel(full_data)
    features = full_data[[
        "hotel_id",
        "accommadation_type_name",
        "hotel_star_rating",
    ]]
    features = pd.get_dummies(features, columns=["accommadation_type_name"])
    labels = full_data["diff_booking_to_cancel"]

    return features, labels


def load_data_to_predict(filename: str):
    full_data = pd.read_csv(filename).drop_duplicates()
    full_data = __add_did_cancel(full_data, True)
    features = full_data[[
        "original_selling_amount",
        "day_of_year", "how_far_in_advance",
        "length_of_stay", "no_of_adults", "hotel_brand_code",
        "customer_nationality", "charge_option",
        'hotel_chain_code', 'original_payment_method'
    ]]
    features = features.fillna(0)
    features["hotel_brand_code"] = features["hotel_brand_code"].rank(method='dense').astype(int)
    features["customer_nationality"] = features["customer_nationality"].rank(method='dense').astype(int)
    features["charge_option"] = features["charge_option"].rank(method='dense').astype(int)
    features["hotel_chain_code"] = features["hotel_chain_code"].rank(method='dense').astype(int)
    features["original_payment_method"] = features["original_payment_method"].rank(method='dense').astype(int)

    return features


def load_classsifier(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # length of stay
    full_data = pd.read_csv(filename).drop_duplicates()
    full_data = __add_did_cancel(full_data)
    features = full_data[[
        "diff_booking_to_cancel",
        "original_selling_amount",
        "day_of_year", "how_far_in_advance",
        "length_of_stay", "no_of_adults", "hotel_brand_code",
        "customer_nationality", "charge_option",
        'hotel_chain_code', 'original_payment_method', "canceled"
    ]]
    features = features.dropna()
    features["hotel_brand_code"] = features["hotel_brand_code"].rank(method='dense').astype(int)
    features["customer_nationality"] = features["customer_nationality"].rank(method='dense').astype(int)
    features["charge_option"] = features["charge_option"].rank(method='dense').astype(int)
    features["hotel_chain_code"] = features["hotel_chain_code"].rank(method='dense').astype(int)
    features["original_payment_method"] = features["original_payment_method"].rank(method='dense').astype(int)
    labels = features["canceled"]

    return features.drop(["canceled", "diff_booking_to_cancel"], axis=1), labels


def load_regression(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # length of stay
    full_data = pd.read_csv(filename).drop_duplicates()
    full_data = __add_did_cancel(full_data)
    features = full_data[[
        "diff_booking_to_cancel",
        "original_selling_amount",
        "day_of_year", "how_far_in_advance",
        "length_of_stay", "no_of_adults", "hotel_brand_code",
        "customer_nationality", "charge_option",
        'hotel_chain_code', 'original_payment_method'
    ]]
    features = features.dropna()
    features["hotel_brand_code"] = features["hotel_brand_code"].rank(method='dense').astype(int)
    features["customer_nationality"] = features["customer_nationality"].rank(method='dense').astype(int)
    features["charge_option"] = features["charge_option"].rank(method='dense').astype(int)
    features["hotel_chain_code"] = features["hotel_chain_code"].rank(method='dense').astype(int)
    features["original_payment_method"] = features["original_payment_method"].rank(method='dense').astype(int)
    labels = features["diff_booking_to_cancel"]

    return features.drop(["diff_booking_to_cancel"], axis=1), labels


def __get_day_of_year(full_data):
    date_list = full_data.booking_datetime.split(" ")[0].split("-")
    year = date_list[0]
    month = date_list[1]
    day = date_list[2]
    day_of_year = datetime.date(int(year), int(month), int(day)).timetuple().tm_yday
    return day_of_year


def __how_far_in_advance(full_data):
    date_list = full_data.booking_datetime.split(" ")[0].split("-")
    year = date_list[0]
    month = date_list[1]
    day = date_list[2]
    booking = datetime.date(int(year), int(month), int(day))
    checking = full_data.checkout_date.split(" ")[0].split("-")
    year = checking[0]
    month = checking[1]
    day = checking[2]
    checkin = datetime.date(int(year), int(month), int(day))
    return abs((booking - checkin).days)


def __length_of_stay(full_data):
    date_list = full_data.checkin_date.split(" ")[0].split("-")
    year = date_list[0]
    month = date_list[1]
    day = date_list[2]
    booking = datetime.date(int(year), int(month), int(day))
    checking = full_data.checkout_date.split(" ")[0].split("-")
    year = checking[0]
    month = checking[1]
    day = checking[2]
    checkin = datetime.date(int(year), int(month), int(day))
    return abs((booking - checkin).days)


def __add_did_cancel(full_data, pred=False):
    pd.options.mode.chained_assignment = None
    canceled = list()
    diff_booking_to_cancel = list()
    day_of_year = list()
    how_far_in_advance = list()
    length_of_stay = list()
    pay_now = list()

    for i in tqdm(range(len(full_data))):
        if not pred:
            if str(full_data.cancellation_datetime.iloc[i]) == "nan":
                diff_booking_to_cancel.append(-1)
                canceled.append(0)
            else:
                diff_booking_to_cancel.append(
                    __diff_of_date_start(full_data.iloc[i][["checkout_date", "cancellation_datetime"]]))
                canceled.append(1)
        if full_data.charge_option.iloc[i] == "Pay Now":
            pay_now.append(1)
        else:
            pay_now.append(0)
        day_of_year.append(__get_day_of_year(full_data.iloc[i]))
        how_far_in_advance.append(__how_far_in_advance(full_data.iloc[i]))
        length_of_stay.append(__length_of_stay(full_data.iloc[i]))

    if not pred:
        full_data["canceled"] = canceled
        full_data["diff_booking_to_cancel"] = diff_booking_to_cancel
    full_data["day_of_year"] = day_of_year
    full_data["how_far_in_advance"] = how_far_in_advance
    full_data["length_of_stay"] = length_of_stay
    full_data["pay_now"] = pay_now
    return full_data


def __diff_of_date_start(date_string):
    date_list = date_string.checkout_date.split(" ")[0].split("-")
    year = date_list[0]
    month = date_list[1]
    day = date_list[2]
    date_booking = datetime.date(int(year), int(month), int(day))
    date_cancel_list = date_string.cancellation_datetime.split(" ")[0].split("-")
    year = date_cancel_list[0]
    month = date_cancel_list[1]
    day = date_cancel_list[2]
    date_cancel = datetime.date(int(year), int(month), int(day))
    return int(abs((date_booking - date_cancel).days))


def __add_time_between_booking_to_cancel(full_data):
    booking_time_df = pd.DataFrame(full_data[["checkout_date", "cancellation_datetime"]])
    full_data["diff_booking_to_cancel"] = booking_time_df.apply(__diff_of_date_start, axis=1)
    return full_data


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """

    pred_y = estimator.predict(X)
    X = X.reset_index()
    for i, y in enumerate(pred_y):
        if BEGIN_CHECK_DATE.timetuple().tm_yday-1 <= X.iloc[i]["day_of_year"] + y <= END_CHECK_DATE.timetuple().tm_yday+1:
            pred_y[i] = 1
        else:
            pred_y[i] = 0

    pd.DataFrame(pred_y, columns=["predicted_values"]).to_csv(filename, index=False)


def load_from_csv():
    df = pd.read_csv("../datasets/Agoda/load2_df.csv")
    cancellation_labels = pd.read_csv("../datasets/Agoda/load2_cancellation_labels.csv")
    cancellation_labels_reg = pd.read_csv("../datasets/Agoda/load2_cancellation_labels_reg.csv")
    return df, cancellation_labels, cancellation_labels_reg


def save_estimator(obj, filename="estimator.pkl"):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_estimator(filename="estimator.pkl"):
    with open(filename, 'rb') as inp:
        estimator = pickle.load(inp)
    return estimator

if __name__ == '__main__':
    np.random.seed(0)
    estimator = load_estimator()
    test_X_sol = load_data_to_predict("../datasets/Agoda/test_set_week_1.csv")
    evaluate_and_export(estimator, test_X_sol, "318166535_315146217_204798383.csv")
    # Load data
    # df, cancellation_labels = load_data2("../datasets/agoda_cancellation_train.csv")
    df, cancellation_labels, cancellation_labels_reg = load_from_csv()
    cancellation_labels = cancellation_labels.rename(columns={"canceled": "y_classifier"})
    cancellation_labels_reg = cancellation_labels_reg.rename(columns={"canceled": "y_regressor"})
    cancellation_labels = pd.concat([cancellation_labels, cancellation_labels_reg], ignore_index=True, sort=False,
                                    axis=1)
    cancellation_labels = cancellation_labels.rename(columns={0: "y_classifier", 1: "y_regressor"})
    # cancellation_labels = pd.merge(cancellation_labels, cancellation_labels_reg, on=["y_classifier", "y_regressor"])
    train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels, test_size=0.2)

    # check most important features and decide which one to use
    # plot_feature_importance(train_X, train_y)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    pred_y = estimator.predict(test_X)
    pred_y_classifier = estimator._predict_classifier(test_X)
    print(estimator.get_classifier_params())
    print(estimator.get_regressor_params())

    test_y_classifier = test_y.y_classifier.to_numpy()
    test_y_regressor = test_y.y_regressor.to_numpy()
    print('clas MAE: ', mean_absolute_error(test_y_classifier, pred_y_classifier))
    print('clas MSE: ', mean_squared_error(test_y_classifier, pred_y_classifier))
    print('MAE: ', mean_absolute_error(test_y_regressor, pred_y))
    print('MSE: ', mean_squared_error(test_y_regressor, pred_y))

    # Store model predictions over test set
    save_estimator(estimator)
    test_X_sol = load_data_to_predict("../datasets/Agoda/test_set_week_1.csv")
    evaluate_and_export(estimator, test_X_sol, "318166535_315146217_204798383.csv")
