import rasterio
import rasterio.mask
import os
import numpy as np
import fiona
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tqdm import tqdm
from rasterio.mask import mask

from prophet import Prophet
from sklearn.metrics import (
    mean_pinball_loss,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    precision_recall_curve,
)
import warnings

# Ignore specific warning categories or all
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Default plotting parameters
font = {"size": 30}
matplotlib.rc("font", **font)


def collect_data(Climate, years, LC_feature, Elv, LC, features, verbose=False):
    """Collects data

    Parameters:
    --------
        Climate: Dict
            Climate data
        years: List[int]
            Years used
        Elv: 2d numpy array
            Elevation data
        LC_feature: Dict
            Land cover historical feature
        LC: Dict
            Land cover target feature
        features: List[str]
            Feature names used in model
        verbose: Optional[bool]
            Additional printable info

    Returns:
    --------
        X: [2d numpy array]
            All the instances with attributes
        y: [1d numpy array]
            Label of each instance
    """

    # Sort features to use for X
    features_clim = []
    flag_lc = False  # if 'lc' feature is in list
    flag_elv = False  # if 'lc' feature is in list

    for f in features:
        if "bio" in f:
            features_clim.append(f)
        elif "lc" in f:
            flag_lc = True
        elif "elv" in f:
            flag_elv = True

    # convert dict into np.arrays
    LC_feature_array = np.stack(
        [LC_feature[year] for year in LC_feature.keys()], axis=-1
    )
    LC_array = np.stack([LC[year] for year in LC.keys()], axis=-1)

    # get raster width and height
    h = LC_array.shape[0]
    w = LC_array.shape[1]

    # flatten years
    LC_feature_array = LC_feature_array.reshape(-1, 1)
    LC_array = LC_array.reshape(-1)
    Elv = Elv.reshape(-1, 1)
    for var in features_clim:
        Climate[var] = Climate[var].reshape(-1)

    Climate_array = np.stack([Climate[var] for var in features_clim], axis=-1)

    if flag_lc and flag_elv:
        X = np.hstack((Climate_array, np.tile(Elv, (len(years), 1)), LC_feature_array))
    elif flag_elv:
        X = np.hstack((Climate_array, np.tile(Elv, (len(years), 1))))
    elif flag_lc:
        X = np.hstack((Climate_array, LC_feature_array))
    else:
        X = Climate_array
    y = LC_array

    if verbose:
        print("X shape:", X.shape)
        print("Y shape:", y.shape)
    return X, y, h, w


def xgbc_scores_binary(clf, X_test, y_test, baseline=False):
    """Evaluates scores of the binary classification model

    Parameters:
    --------
        clf: xgbc model
            Prefitted model
        X_test: 2d numpy array
            All the instances with attributes
        y_test: List[int]
            Label of each instance
        baseline: Optional[Bool]
            Flag weather scores are baseline

    Returns:
    --------
        scores: Dict[str, float]
    """
    pred_proba = clf.predict_proba(X_test)[::, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, pred_proba)
    # locate the index of the largest f score
    fscore = (2 * precision * recall) / (precision + recall + 0.00001)
    ix = np.argmax(fscore)
    scores = dict()

    if baseline == False:
        prediction = [1 if x >= thresholds[ix] else 0 for x in pred_proba]
        scores["RA_score"] = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        scores["threshold"] = thresholds[ix]
    else:
        prediction = X_test[:, -1]

    scores["recall_score"] = recall_score(y_test, prediction, average="binary")
    scores["precision_score"] = precision_score(y_test, prediction, average="binary")
    scores["balanced_accuracy_score"] = balanced_accuracy_score(y_test, prediction)
    return scores


def xgbc_scores_multi(clf, X_test, y_test):
    """Evaluates scores of the multi classification model

    Parameters:
    --------
        clf: xgbc model
            Prefitted model
        X_test: [2d numpy array]
            All the instances with attributes
        y_test: List[int]
            Label of each instance

    Returns:
    --------
        scores: Dict[str, float]
    """
    prediction = clf.predict(X_test)
    scores = dict()
    scores["balanced_accuracy_score"] = balanced_accuracy_score(y_test, prediction)
    scores["RA_score"] = roc_auc_score(
        y_test, clf.predict_proba(X_test), multi_class="ovr"
    )
    scores["recall_score"] = recall_score(y_test, prediction, average="weighted")
    scores["precision_score"] = precision_score(y_test, prediction, average="weighted")
    return scores


def array2raster(file_name, array, path):
    """Saves GTiff file from numpy.array

    Parameters:
    --------
        file_name: str
            File name to use for (re)writing
        array: 2d numpy array
            Data to write
        path: str
            Path to the folder with source GeoTiff file

    Returns:
    --------
        .tif file
    """
    # Random source file to copy geodata for creating GeoTiff
    fn = os.listdir(os.path.join(path, "SEAsia", "Climate_future"))[-1]
    template = rasterio.open(os.path.join(path, "SEAsia", "Climate_future", fn))

    dtype = array.dtype
    with rasterio.open(file_name, "w", **template.meta) as dest:
        dest.write(array.astype(dtype), 1)


def avg_future_pred(Climate_future, years_future, LC_feature, Elv, LC, clf, features):
    """Calculates average future predictions

    Parameters:
    --------
        Climate_future: Dict
            Future climate data
        years_future: List[int]
            Years included in Climate_future
        LC_features: Dict
            Features of the LC model
        Elv: ndarray
            Elevation data
        LC: Dict
            Land cover data
        clf: xgbc model
            Prefitted model
        features: list[str]
            Features of the model

    Returns:
    --------
        prediction_prob: [1d numpy array]
            Future probability prediction
    """
    CMIPs = Climate_future.keys()
    prediction_stack = np.empty([0, Elv.shape[0] * Elv.shape[1]])

    # Loop through each CMIP simulation
    for item in CMIPs:
        # Collect data for current simulation
        X_future, y_test, h, w = collect_data(
            Climate_future[item],
            years_future,
            LC_feature,
            Elv,
            LC,
            features,
            verbose=False,
        )

        # Make probability prediction for current simulation
        prediction_prob = clf.predict_proba(X_future)[:, 1]

        # Add current prediction to prediction stack
        prediction_stack = np.vstack((prediction_stack, prediction_prob))

    # Calculate average probability prediction across all simulations
    prediction_prob_avg = np.mean(prediction_stack, axis=0).reshape(-1)

    return prediction_prob_avg, y_test, h, w


def changes2raster(
    clf,
    model_name,
    features,
    Elv,
    LC_feature,
    Climate,
    LC,
    years,
    year_current,
    path,
    path_pics,
    thresh,
):
    """Plots on the map coloured pixels showing positive/ negative changes
       Compared with LC baseline

    Parameters:
    --------
        clf: XGBClassifier model
            Prefitted model
        model_name: str
            Model name
        features: List[str]
            List of features used in model
        Elv: 2d numpy array
            Elevation data
        LC_feature: Dict
            Land cover historical feature
        Climate: Dict
            Climate data
        LC: Dict
            Land cover target feature
        years: List
            Used years
        year_current: int
            Current year
        path: str
            Path to save the .tif with misclassified areas
        path_pics: str
            Path to save the new pic
        thresh: float
            Threshold to define positive/negative change

    Returns:
    --------
        three .tif files
    """
    # Compose dataset for the model
    prediction_prob, y_test, h, w = avg_future_pred(
        Climate, years, LC_feature, Elv, LC, clf, features
    )

    # See the difference between prediction and the baseline
    prediction = [1 if x >= thresh else 0 for x in prediction_prob]
    y_proba = prediction_prob - y_test
    # Creating the masks
    # Creating the masks
    mask1 = (y_test == 1)
    mask2 = (prediction_prob < 0.5)
    neg_trend_count = (mask1 & mask2).sum()
    if mask1.sum() != 0:
        neg_trend = neg_trend_count / mask1.sum() * 100
        print(f"Proportion of negative trends where y_test is 1 and prediction_prob < 0.5: {neg_trend} %")
    else:
        print("No elements in y_test are 1, cannot compute proportion.")
    y = prediction - y_test
    print(np.unique(y, return_counts=True)) 
    ####################################################
    # Reshape to correlate with sourse rasters
    y_proba_rect = y_proba.reshape(h, w)
    y_rect = y.reshape(h, w)
    y_test = y_test.reshape(h, w)

    # Check if the folder for record exists
    folder = os.path.join(path_pics, model_name)
    os.makedirs(folder, exist_ok=True)

    # Save as tif
    year = str(year_current)
    array2raster(
        os.path.join(folder, f"proba_{year}.tif"), prediction_prob.reshape(h, w), path
    )
    array2raster(os.path.join(folder, f"changes_proba_{year}.tif"), y_proba_rect, path)
    array2raster(os.path.join(folder, f"changes_{year}.tif"), y_rect, path)
    array2raster(os.path.join(path_pics, "basic.tif"), y_test, path)


def crop_raster(file_name, source_tif, count, shapes, path):
    """Crops .tif file

    Parameters:
    --------
        file_name: str
            New file name
        source_tif: str
            Original tif file to crop
        count: int
            Number of bands
        shapes:
            Array of shapes defining country border
        path: str
            Path to the folder with source GeoTiff file

    Returns:
    --------
        Saves the results to .tif file
    """
    # Random source file to copy geodata for creating GeoTiff
    fn = os.listdir(os.path.join(path, "SEAsia", "Climate_future", ""))[-1]
    template = rasterio.open(os.path.join(path, "SEAsia", "Climate_future", fn))

    # Crop the tif
    with rasterio.open(source_tif) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = template.meta

    out_meta.update(
        {
            "driver": "GTiff",
            "count": count,
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    # Write new tif
    with rasterio.open(file_name, "w", **out_meta) as dest:
        dest.write(out_image)


def crop_country(model_name, prefixes, country_names, path, path_pics, year):
    """Crops .tifs with changes up to country

    Parameters:
    --------
        model_name: str
            Model name as it is spelled in corresponding folder
        prefixes: List[str]
            Short country names used in shapefiles
        country_name: List[str]
            Normal spelling of country
        path: str
            Path to the folder with source GeoTiff file
        path_pics: str
            Path to save the new pic
        year: Int
            Year to plot

    Returns:
    --------
        Saves the results to 3.tif files for every country
    """
    for prefix, country_name in zip(prefixes, country_names):
        # Load administrative borders
        with fiona.open(
            os.path.join(path, "boundary", f"gadm41_{prefix}_0.shx"), "r"
        ) as sf:
            shapes = [feature["geometry"] for feature in sf]

        # Crop the country pic with classes existed in historical year
        crop_raster(
            os.path.join(path_pics, f"basic_{country_name}.tif"),
            os.path.join(path_pics, "basic.tif"),
            1,
            shapes,
            path,
        )

        # Check if the folder for record exists
        folder = os.path.join(path_pics, model_name)
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        # Crop the changes picture
        year = str(year)
        crop_raster(
            os.path.join(folder, f"changes_proba_{year}_{country_name}.tif"),
            os.path.join(folder, f"changes_proba_{year}.tif"),
            1,
            shapes,
            path,
        )
        crop_raster(
            os.path.join(folder, f"changes_{year}_{country_name}.tif"),
            os.path.join(folder, f"changes_{year}.tif"),
            1,
            shapes,
            path,
        )


def yield_rolling_mean(Production, prefixes, years, **kwargs):
    """Yields rolling mean of Y_train and Y_test

    Parameters:
    --------
        Production: ndarray
            Production data for year, crop, country
        prefixes: List[str]
            Short country names
        years: List[int]
            Years to plot
    Returns:
    --------
        Yield: Dict
            Dictionary with rolling mean of Y_train and Y_test
    """
    # Rolling mean window parameter
    window = kwargs.get("window", 2)

    plt.figure(figsize=(30, 12))
    Yield = dict.fromkeys(prefixes)

    # Country data for rice
    for prefix in prefixes:
        c = np.random.rand(3)
        Y_prefix = Production[
            (Production[:, 2] == prefix) & (Production[:, 4] == "Rice")
        ]
        # years available
        years_country = pd.Series(list(map(int, Y_prefix[:, 3])))
        # yield available
        yield_country = pd.Series(list(map(float, Y_prefix[:, 0])), index=years_country)
        # Pad data to avoid NaNs in rolling mean
        yield_country = yield_country.reindex(
            range(min(years_country), max(years_country) + 1), method="nearest"
        )
        yield_country = yield_country.rolling(
            window=window, center=True, min_periods=1
        ).mean()
        if prefix in ["KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "THA", "VNM"]:
            plt.plot(yield_country.index, yield_country, label=prefix, color=c)
        Yield[prefix] = yield_country
    plt.legend(bbox_to_anchor=(1.04, 0.83), loc="center left")
    plt.suptitle("Rice yield")

    return Yield


def fert_forecast(
    Fert,
    Total_area,
    prefixes,
    country_names,
    year_rice,
    method="original",
    test_period=2,
    n_year_rice=10,
):
    """
    Plots fertilizers forecast using specified method: original, AR, ARIMA, Holt-Winters, VAR, or Prophet.
    Includes metrics RMSE and MAE, and provides hyperparameters for each method.

    Parameters:
    -----------
    Fert: Dict
        Fert data for year, type, country
    Total_area: Dict
        Total arable area data for year, country
    prefixes: List[str]
        Short country names
    country_names: List[str]
        Full country names
    year_rice: List[int]
        Years listed in FAOSTAT data
    method: str
        Forecasting method
    test_period: int
        Number of years to use for the test set
    n_year_rice: int
        Number of years to forecast

    Returns:
    --------
    Dict of forecasted fertilizer data and a dict of average metrics.
    """

    year_future_fert = np.arange(year_rice[-1] + 1, year_rice[-1] + n_year_rice + 1)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
    Fert_future = {key: [] for key in prefixes}
    metrics = {
        nutrient: {"MAPE": [], "MAE": [], "RMSE": []}
        for nutrient in ["N", "P2O5", "K2O"]
    }

    for prefix, name in zip(prefixes, country_names):
        values_N, values_P2O5, values_K2O = [], [], []
        for year in year_rice:
            fert_data = Fert.get(prefix, {}).get(year, [np.nan, np.nan, np.nan])
            fert_data += [np.nan] * (3 - len(fert_data))

            if year in Total_area[prefix]:
                area = Total_area[prefix][year]["total"]
            else:
                area = np.nan

            values_N.append(float(fert_data[0]) / area if area else np.nan)
            values_P2O5.append(float(fert_data[1]) / area if area else np.nan)
            values_K2O.append(float(fert_data[2]) / area if area else np.nan)
        df_var = pd.DataFrame({"N": values_N, "P2O5": values_P2O5, "K2O": values_K2O})
        c = np.random.rand(3)
        for idx, (values, nutrient) in enumerate(
            zip([values_N, values_P2O5, values_K2O], ["N", "P2O5", "K2O"])
        ):
            values = pd.Series(values).bfill().values
            train, test = values[:-test_period], values[-test_period:]
            forecast_test = None

            if method == "original":
                diff = values[-1] - values[-n_year_rice]
                forecast = np.array(values[-n_year_rice:]) + diff
            else:
                if method == "AR":
                    model = AutoReg(train, lags=2)
                elif method == "ARIMA":
                    model = ARIMA(train, order=(1, 1, 1))
                elif method == "SARIMA":
                    model = SARIMAX(
                        train,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 10),
                    )
                elif method == "Holt-Winters":
                    model = ExponentialSmoothing(
                        train,
                        trend="add",
                        seasonal="multiplicative",
                        seasonal_periods=min(7, len(train) // 2),
                        dates=pd.to_datetime(
                            [f"{year}-12-31" for year in year_rice[:-test_period]]
                        ),
                        freq="YE",
                    )
                elif method == "VAR" and idx == 0:
                    model = VAR(df_var[:-test_period])

                if method not in ["Prophet", "VAR", "SARIMA"]:
                    model_fit = model.fit()
                    forecast_test = model_fit.forecast(steps=test_period)

                if method == "SARIMA":
                    model_fit = model.fit(disp=-1)
                    forecast_test = model_fit.forecast(steps=test_period)

                if method == "VAR":
                    model_fit = model.fit()
                    forecast_test = model_fit.forecast(
                        y=df_var[:-test_period].values, steps=test_period
                    )[:, idx]

                if method == "Prophet":
                    df_prophet = pd.DataFrame(
                        {
                            "ds": pd.date_range(
                                start=str(year_rice[0]),
                                periods=len(year_rice),
                                freq="YE",
                            ),
                            "y": values,
                        }
                    )
                    model = Prophet()
                    model.fit(df_prophet[:-test_period])
                    future = model.make_future_dataframe(periods=test_period, freq="YE")
                    forecast_test = model.predict(future)["yhat"][-test_period:].values

                if forecast_test is not None:
                    metrics[nutrient]["MAPE"].append(
                        mean_absolute_percentage_error(test, forecast_test)
                    )
                    metrics[nutrient]["MAE"].append(
                        mean_absolute_error(test, forecast_test)
                    )
                    metrics[nutrient]["RMSE"].append(
                        np.sqrt(mean_squared_error(test, forecast_test))
                    )

                # Retrain on full data for final forecast
                if method == "AR":
                    model = AutoReg(values, lags=2)
                elif method == "ARIMA":
                    model = ARIMA(values, order=(1, 1, 1))
                elif method == "SARIMA":
                    model = SARIMAX(
                        values,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 10),
                    )
                elif method == "Holt-Winters":
                    model = ExponentialSmoothing(
                        values,
                        trend="add",
                        seasonal="multiplicative",
                        seasonal_periods=min(7, len(train) // 2),
                        dates=pd.to_datetime([f"{year}-12-31" for year in year_rice]),
                        freq="YE",
                    )
                elif method == "VAR" and idx == 0:
                    model = VAR(df_var)
                if (
                    method == "VAR" and idx == 0
                ):  # VAR is a multivariate model, run it once
                    model_fit = model.fit()
                    forecast = model_fit.forecast(y=df_var.values, steps=n_year_rice)[
                        :, idx
                    ]
                elif method == "Prophet":
                    future = model.make_future_dataframe(periods=n_year_rice, freq="YE")
                    forecast = model.predict(future)["yhat"][-n_year_rice:].values
                elif method not in ["VAR", "Prophet"]:
                    model_fit = model.fit(disp=-1)
                    forecast = model_fit.forecast(steps=n_year_rice)

            if forecast is not None:
                forecast[forecast < 0] = 0
                combined_year_rice = np.concatenate((year_rice[-1:], year_future_fert))
                combined_forecast = np.concatenate(
                    ([values[-1]], np.where(forecast < 0, 0, forecast))
                )
                if name in ['Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar', 'Philippines', 'Thailand', 'Viet Nam']:
                    
                    ax[idx].plot(year_rice, values, label=name, color=c, linewidth=4)
                    ax[idx].plot(
                        combined_year_rice,
                        combined_forecast,
                        marker=".",
                        color=c,
                        linestyle="--",
                        linewidth=4,
                    )
                Fert_future[prefix].append(forecast.tolist())

    metrics_avg = {
        nutrient: {
            metric: np.mean(values) for metric, values in metrics[nutrient].items()
        }
        for nutrient in metrics
    }

    for i, title in enumerate(["(a)", "(b)", "(c)"]):  # ["N", "P2O5", "K2O"]):
        ax[i].set_title(title, fontweight="bold", fontsize=50)
        ax[i].set_xlabel("Year", fontsize=50)
        ax[i].set_ylabel("kg/ha", fontsize=50)
        if i == 1:
            ax[i].legend(bbox_to_anchor=(0.58, 0.75), ncol=2, fontsize="large")
        ax[i].tick_params(axis="x", labelsize=44)
        ax[i].tick_params(axis="y", labelsize=44)

    plt.tight_layout()
    plt.savefig("/app/data/03-results/pics/fert_forecast.jpg", bbox_inches="tight", dpi=300)

    print(metrics_avg)
    print(
        "Mean MAPE:",
        sum(value["MAPE"] for value in metrics_avg.values()) / len(metrics_avg),
    )
    print(
        "Mean RMSE:",
        sum(value["RMSE"] for value in metrics_avg.values()) / len(metrics_avg),
    )
    return Fert_future, metrics_avg, df_var, metrics


# def fert_forecast_old(Fert, Total_area,
#                   prefixes, country_names,
#                   years, **kwargs):
#     """Plots fertilezers forecast

#     Parameters:
#     --------
#         Fert: Dict
#             Fert data for year, type, country
#         Total_area: Dict
#             Total arable area data for year, country
#         prefixes: List[str]
#             Short country names
#         country_names: List[str]
#             Normal spelling of country
#         years: List[int]
#             Years listed in FAOSTAT data
#     Returns:
#     --------
#         Fert_future: Dict
#     """
#     n_years = kwargs.get('n_years', 10)
#     years_future_fert = np.arange(years[-1], years[-1]+n_years)

#     fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
#     Fert_future = {keys: [] for keys in prefixes}

#     for prefix, name in zip(prefixes, country_names):
#         values_N, values_P205, values_K2O = [], [], []
#         c = np.random.rand(3,)
#         for year in years:
#             values_N.append(float(Fert[prefix][year][0])/Total_area[prefix][year]['total'])
#             values_P205.append(float(Fert[prefix][year][1])/Total_area[prefix][year]['total'])
#             values_K2O.append(float(Fert[prefix][year][2])/Total_area[prefix][year]['total'])

#         values_N_diff = values_N[-1] - values_N[-n_years]
#         values_N_future = np.copy(values_N[-n_years:]) + values_N_diff
#         ax[0].plot(years, values_N, label=name, color=c, linewidth=3)
#         ax[0].plot(years_future_fert, values_N_future,
#                    marker='.', color=c, linestyle='--', linewidth=3)
#         Fert_future[prefix].append(values_N_future)

#         values_P205_diff = values_P205[-1] - values_P205[-n_years]
#         values_P2O5_future = np.copy(values_P205[-n_years:]) + values_P205_diff
#         ax[1].plot(years, values_P205, label=name, color=c, linewidth=3)
#         ax[1].plot(years_future_fert, values_P2O5_future,
#                    marker='.', color=c, linestyle='--', linewidth=3)
#         Fert_future[prefix].append(values_P2O5_future)

#         values_K2O_diff = values_K2O[-1] - values_K2O[-n_years]
#         values_K2O_future = np.copy(values_K2O[-n_years:]) + values_K2O_diff
#         ax[2].plot(years, values_K2O, label=name, color=c, linewidth=3)
#         ax[2].plot(years_future_fert, values_K2O_future,
#                    marker='.', color=c, linestyle='--', linewidth=3)
#         Fert_future[prefix].append(values_K2O_future)

#     ax[0].set_title('N')
#     ax[1].set_title('P2O5')
#     ax[2].set_title('K2O')

#     for i, ax in enumerate(fig.axes):
#         ax.set_xticklabels(ax.get_xticklabels(), fontsize=44)
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=44)
#         for label in ax.get_xticklabels()[::2]:
#             label.set_visible(False)
#         for label in ax.get_yticklabels()[::2]:
#             label.set_visible(False)

#         ax.set_xlabel('Year', fontsize=50)
#         ax.set_ylabel('kg/ha', fontsize=50)

#     plt.legend(bbox_to_anchor=(0.35, 0.75), ncol=2)
#     plt.savefig("fert.jpg", bbox_inches='tight', dpi=300)
#     return Fert_future


def collect_data_yield(
    path_faostat,
    data_production,
    prefixes,
    country_names,
    Climate,
    Fert,
    Yield,
    Total_area,
    n_years_test,
    crops,
    vars,
    years,
    n_feat,
):
    """
    Collects data from FAOSTAT and CMIP5

    Parameters:
    --------
        path_faostat: str
            Path to the folder with faostat data
        data_production: str
            Production data file name
        prefixes: List[str]
            Short country names used in shapefiles
        country_name: List[str]
            Normal spelling of country
        Climate: dict
            Climate data
        Fert: dict
            Fertilizer data
        Yield: dict
            Yield data
        Total_area: dict
            Total area of each country
        crops: List[str]
            List of crops
        years: List[int]
            Years to use
    Returns:
    --------
    X_train, Y_train, X_test, Y_test
    """
    Y_train = Y_test = np.zeros((0, 5))
    X_train = X_test = np.zeros((0, n_feat))

    # Empty dictionaries for including in feature space
    Const_country = {keys: 0 for keys in prefixes}
    Const_crop = {keys: 0 for keys in crops}
    Trend = {keys: 0 for keys in prefixes}
    years = years[years > 1995]
    
    reader = csv.reader(open(os.path.join(path_faostat, data_production), "r"))

    for data in reader:
        country_prod = data[0]
        crop_prod = data[2]
        year_prod = int(data[3])
        const_0 = Trend.copy()

        for prefix, country_name in zip(prefixes, country_names):
            if (
                (country_name in country_prod)
                & (year_prod <= years[-1])
                & (year_prod >= years[1])
            ):
                climate = Climate[prefix][year_prod]
                climate_array = np.stack(
                    [climate[var] for var in vars], axis=1
                ).reshape(1, -1, order="F")

                fert_array = [
                    x / Total_area[prefix][year_prod]["total"]
                    for x in Fert[prefix][year_prod]
                ]
                crop_area = Total_area[prefix][year_prod]["Rice"]
                time_trend = year_prod - years[0]

                const_0 = Trend.copy()
                const_0[prefix] = time_trend
                const_array_0 = np.stack(
                    [const_0[prefix] for prefix in prefixes], axis=-1
                ).reshape(1, -1)

                const_1 = Const_country.copy()
                const_1[prefix] = 1
                const_array_1 = np.stack(
                    [const_1[prefix] for prefix in prefixes], axis=-1
                ).reshape(1, -1)

                const_2 = Const_crop.copy()
                const_2[crop_prod] = 1
                const_array_2 = np.stack(
                    [const_2[crop] for crop in crops], axis=-1
                ).reshape(1, -1)
                array = np.hstack(
                    (
                        climate_array[0],
                        fert_array,
                        const_array_0[0],
                        const_array_1[0],
                        const_array_2[0],
                    )
                ).astype(float)
                if any(year_prod == year for year in years[:-n_years_test]):
                    Y_train = np.vstack(
                        (
                            Y_train,
                            [
                                Yield[prefix][year_prod],
                                Yield[prefix][year_prod] * crop_area,
                                prefix,
                                year_prod,
                                crop_prod,
                            ],
                        )
                    )
                    X_train = np.vstack((X_train, array))

                # last years for test
                elif any(year_prod == year for year in years[-n_years_test:]):
                    Y_test = np.vstack(
                        (
                            Y_test,
                            [
                                Yield[prefix][year_prod],
                                Yield[prefix][year_prod] * crop_area,
                                prefix,
                                year_prod,
                                crop_prod,
                            ],
                        )
                    )
                    X_test = np.vstack((X_test, array))

    return X_train, Y_train, X_test, Y_test


def collect_future_data(
    path_faostat,
    data_production,
    prefixes,
    country_names,
    Climate,
    CMIPs,
    Fert_future,
    crops,
    vars,
    years,
    years_future,
    n_feat,
):
    """
    Collects data from FAOSTAT and CMIP5

    Parameters:
    --------
        path_faostat: str
            Path to the folder with faostat data
        data_production: str
            Production data file name
        prefixes: List[str]
            Short country names used in shapefiles
        country_name: List[str]
            Normal spelling of country
        Climate: dict
            Climate data
        CMIPs: list
            CMIP models names
        Fert_future: dict
            Fertilizer data
        Total_area: dict
            Total area of each country
        crops: List[str]
            List of crops
        years: List[int]
            Years to use
        years_future: List[int]
            Years to use for future
    Returns:
    --------
    X_future

    """

    X_future_models = {key: [] for key in CMIPs}
    for model in CMIPs:
        X_future = np.zeros((0, n_feat))
        # Empty dictionaries for including in feature space
        Const_country = {keys: 0 for keys in prefixes}
        Const_crop = {keys: 0 for keys in crops}
        Trend = {keys: 0 for keys in prefixes}

        reader = csv.reader(open(os.path.join(path_faostat, data_production), "r"))
        for data in reader:

            if int(data[3]) == years[-1]:
                country_prod = data[0]
                crop_prod = data[2]

                for prefix, country_name in zip(prefixes, country_names):
                    if country_name in country_prod:

                        climate = Climate[prefix][years_future[0]][model]
                        climate_array = np.stack(
                            [climate[var] for var in vars], axis=1
                        ).reshape(1, -1, order="F")

                        fert_index = years_future[0] - years[-1]
                        fert_array = [
                            Fert_future[0][prefix][0][fert_index],
                            Fert_future[0][prefix][1][fert_index],
                            Fert_future[0][prefix][2][fert_index],
                        ]

                        time_trend = years_future[0] - years[0]
                        const_0 = Trend.copy()
                        const_0[prefix] = time_trend
                        const_array_0 = np.stack(
                            [const_0[prefix] for prefix in prefixes], axis=-1
                        ).reshape(1, -1)

                        const_1 = Const_country.copy()
                        const_1[prefix] = 1
                        const_array_1 = np.stack(
                            [const_1[prefix] for prefix in prefixes], axis=-1
                        ).reshape(1, -1)

                        const_2 = Const_crop.copy()
                        const_2[crop_prod] = 1
                        const_array_2 = np.stack(
                            [const_2[crop] for crop in crops], axis=-1
                        ).reshape(1, -1)

                        array = np.hstack(
                            (
                                climate_array[0],
                                fert_array,
                                const_array_0[0],
                                const_array_1[0],
                                const_array_2[0],
                            )
                        ).astype(float)
                        X_future = np.vstack((X_future, array))

        # Select rice rows only
        X_future = X_future[X_future[:, -1] == 1][:, :-4]
        X_future_models[model] = X_future

    return X_future_models


def yield_outcome(
    yield_future,
    Y_test,
    X_future_rice,
    CMIPs,
    Total_area,
    neg_trend,
    prefixes,
    country_names,
    year_base,
):
    """
    Calculate the yield of the crop in the future

    Parameters
    ----------
    Y_test : ndarray
        Test data
    X_future_rice : ndarray
        Future data
    Total_area : dict
        Area data
    neg_trend : dict
        Negative trend data
    prefixes : list(str)
        Prefixes of countries
    country_names : list(str)
        Names of the countries
    year_base : int
        Year of the base year
    """
    columns_df = [
        "Country",
        "Area_reduction",
        "Yield_hist",
        "Yield_future",
        "Yield_ratio",
        "Prod_hist",
        "Prod_future",
        "Prod_ratio",
    ]
    df = pd.DataFrame(columns=columns_df)
    # n_count = len(prefixes)
    yield_future_mean = {key: np.mean(values) for key, values in yield_future.items()}

    for i, (prefix, country) in enumerate(zip(prefixes, country_names)):
        df.loc[i, "Area_reduction"] = -neg_trend[prefix]

        # yield historical
        conditions = (
            (Y_test[:, 3] == str(year_base))
            & (Y_test[:, 2] == prefix)
            & (Y_test[:, 4] == "Rice")
        )
        prod_hist = int(float(Y_test[conditions][0, 1]))
        df.loc[i, "Country"] = country
        df.loc[i, "Yield_hist"] = np.round(float(Y_test[conditions][0, 0]), 2)
        df.loc[i, "Prod_hist"] = prod_hist

        # crop production in future. Average over outcomes of all CMIP models
        # yield_future = []
        # for model in CMIPs:
        #     X_future = X_future_rice[model]
        #     X_future_local = X_future[X_future[:, -n_count+i] == 1]
        #     yield_future.append(modelMid.predict(X_future_local))
        # yield_future = np.mean(yield_future)

        # Bootstraped mean as a forecast
        df.loc[i, "Yield_future"] = np.round(yield_future_mean[prefix], 2)
        prod_future = (
            yield_future_mean[prefix]
            * Total_area[prefix][year_base]["Rice"]
            * (1 - neg_trend[prefix] / 100)
        )
        df.loc[i, "Prod_future"] = int(prod_future)

    df["Yield_ratio"] = (df["Yield_future"] - df["Yield_hist"]) / df["Yield_hist"] * 100
    df["Prod_ratio"] = (df["Prod_future"] - df["Prod_hist"]) / df["Prod_hist"] * 100
    df.Yield_ratio = df["Yield_ratio"].round(decimals=1)
    df.Prod_ratio = df["Prod_ratio"].round(decimals=1)
    return df


def reshape_tiffs(path_src, path_dest, height, width):
    """Reshapes .tiff file and writes new one

    Parameters:
    --------
        path_src: Callable[str]
            Source folder
        path_dest: Callable[Dict]
            Destination folder
        bound: Callable[Object]
            Transformation properties

    Returns:
    --------
        new cropped .tiff files
    """
    files = os.listdir(path_src)
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    # Apply those parameters for transformation
    for fname in tqdm(files):
        filepath = path_src + fname

        with rasterio.open(filepath) as src:
            # Create a new cropped raster to write to
            profile = src.profile
            bbox = [
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top,
            ]
            bbox_size = (height, width)

            transform = rasterio.transform.from_bounds(
                *bbox, width=bbox_size[1], height=bbox_size[0]
            )

            profile.update(
                {"height": bbox_size[0], "width": bbox_size[1], "transform": transform}
            )

            with rasterio.open(path_dest + fname, "w", **profile) as dst:
                # Read the data and write it to the output raster
                dst.write(src.read())
    print(len(files), "images reshaped")


def crop_tiff(file_in, file_out, bound):
    """Crops .tiff file and writes new one

    Parameters:
    --------
        path_src: Callable[str]
            Source folder
        path_dest: Callable[Dict]
            Destination folder
        bound: Callable[Object]
            Transformation properties

    Returns:
    --------
        new cropped .tiff files
    """
    with rasterio.open(file_in) as src:
        out_image, out_transform = mask(src, [bound.poly], crop=True)
        # Create a new cropped raster to write to
        profile = src.profile
        profile.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(file_out, "w", **profile) as dst:
            # Read the data and write it to the output raster
            dst.write(out_image)


def reshape_tiff(file_in, file_out, bound):
    """Reshapes .tiff file and writes new one

    Parameters:
    --------
        path_src: Callable[str]
            Source folder
        path_dest: Callable[Dict]
            Destination folder
        bound: Callable[Object]
            Transformation properties

    Returns:
    --------
        new cropped .tiff files
    """

    with rasterio.open(file_in) as src:
        # Create a new cropped raster to write to
        profile = src.profile
        profile.update(
            {"height": bound.height, "width": bound.width, "transform": bound.transform}
        )

        with rasterio.open(file_out, "w", **profile) as dst:
            # Read the data and write it to the output raster
            dst.write(src.read())


def crop_country_file(
    prefixes, country_names, path, path_pics, file_to_crop="rice_2018"
):
    """Crops .tifs with changes up to country

    Parameters:
    --------
        model_name: str
            Model name as it is spelled in corresponding folder
        prefixes: List[str]
            Short country names used in shapefiles
        country_name: List[str]
            Normal spelling of country
        path: str
            Path to the folder with source GeoTiff file
        path_pics: str
            Path to save the new pic
        file_to_crop: str
            File name within path_pics/rice_climate_lc, not path


    Returns:
    --------
        Saves the results to 3.tif files for every country
    """
    os.makedirs(os.path.join(path_pics, "rice_climate_lc"), exist_ok=True)
    for prefix, country_name in zip(prefixes, country_names):
        # Load administrative borders
        with fiona.open(
            os.path.join(path, "boundary", f"gadm41_{prefix}_0.shx"), "r"
        ) as sf:
            shapes = [feature["geometry"] for feature in sf]

        # Crop the country pic with changes
        file_to_crop_ = (
            file_to_crop.split(".")[0] if "." in file_to_crop else file_to_crop
        )
        crop_raster(
            os.path.join(
                path_pics, "rice_climate_lc", f"{file_to_crop_}_{country_name}.tif"
            ),
            "/app/data/00-raw/Rice_map/out3.tif" if file_to_crop_ == 'rice_2018' else os.path.join(path_pics, "rice_climate_lc", f"{file_to_crop_}.tif"),
            1,
            shapes,
            path,
        )


def meanPinBallLossLower(preds, dmat, alpha=0.05):
    """
    Custom-made mean pin Ball loss
    y_true = [1, 2, 3]
    mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    which retunrs 0.033333333
    """

    targets = dmat.get_label()
    loss = mean_pinball_loss(targets, preds, alpha=alpha)

    return "PinBallLoss", loss


def meanPinBallLossUpper(preds, dmat, alpha=0.95):
    """
    Custom-made mean pin Ball loss
    y_true = [1, 2, 3]
    mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    which retunrs 0.033333333
    """

    targets = dmat.get_label()
    loss = mean_pinball_loss(targets, preds, alpha=alpha)

    return "PinBallLoss", loss


# hard or smooth
whichOne = "smooth"
# https://gist.github.com/Nikolay-Lysenko/06769d701c1d9c9acb9a66f2f9d7a6c7


def log_cosh_quantile(alpha):
    """
    LogCosh quantile is nothing more than a smooth quantile loss function.
    This funciotion is C^oo so C1 and C2 which is all we need.
    """

    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)

        if whichOne == "smooth":
            grad = np.tanh(err)
            hess = 1 / np.cosh(err) ** 2
        if whichOne == "hard":
            grad = np.where(err < 0, alpha, (1 - alpha))
            hess = np.ones_like(err)

        return grad, hess

    return _log_cosh_quantile


def proba_thresholding(
    path_proba="/app/data/03-results/model_climate_lc/changes_proba_2028.tif",
    path_model="/app/data/03-results/model_climate_lc/changes_2028_.tif",
    th_pos=0.5,
    th_neg=0.5,
):

    model_prob = rasterio.open(path_proba).read(1)
    model_change = (model_prob > th_pos) * 1 - (model_prob < -th_neg) * 1
    with rasterio.open(path_proba) as src:
        profile = src.profile

    with rasterio.open(path_model, "w", **profile) as dst:
        # Read the data and write it to the output raster
        dst.write(model_change, 1)
