import rasterio
import rasterio.mask
import os
import numpy as np
import fiona
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_score, precision_recall_curve
import matplotlib

# Default plotting parameters
font = {'size': 30}
matplotlib.rc('font', **font)


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
    flag = False  # if 'lc' feature is in list

    for f in features:
        if ('bio' in f):
            features_clim.append(f)
        elif 'lc' in f:
            flag = True

    # convert dict into np.arrays
    LC_feature_array = np.stack([LC_feature[year] for year in LC_feature.keys()], axis=-1)
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

    if flag:
        X = np.hstack((Climate_array, np.tile(Elv, (len(years), 1)), LC_feature_array))
    else:
        X = np.hstack((Climate_array, np.tile(Elv, (len(years), 1))))
    y = LC_array

    if verbose:
        print('X shape:', X.shape)
        print('Y shape:', y.shape)
    return X, y, h, w


def xgbc_scores_binary(clf, X_test, y_test):
    """Evaluates scores of the binary classification model

    Parameters:
    --------
        clf: xgbc model
            Prefitted model
        X_test: 2d numpy array
            All the instances with attributes
        y_test: List[int]
            Label of each instance

    Returns:
    --------
        scores: Dict[str, float]
    """
    pred_proba = clf.predict_proba(X_test)[::, 1]
    precision, recall, thresholds = precision_recall_curve(y_test,
                                                           pred_proba)
    # locate the index of the largest f score
    fscore = (2 * precision * recall) / (precision + recall+0.00001)
    ix = np.argmax(fscore)

    prediction = [1 if x >= thresholds[ix] else 0 for x in pred_proba]
    scores = dict()

    scores['RA_score'] = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    scores['threshold'] = thresholds[ix]
    scores['recall_score'] = recall_score(y_test, prediction, average='binary')
    scores['precision_score'] = precision_score(y_test, prediction, average='binary')
    scores['balanced_accuracy_score'] = balanced_accuracy_score(y_test, prediction)
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
    scores['balanced_accuracy_score'] = balanced_accuracy_score(y_test, prediction)
    scores['RA_score'] = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
    scores['recall_score'] = recall_score(y_test, prediction, average='weighted')
    scores['precision_score'] = precision_score(y_test, prediction, average='weighted')
    return scores


def array2raster(file_name, array, path):
    """Saves GTiff file from numpy.array

    Parameters:
    --------
        file_name: str
            File name to use for (re)writing, optionally with path
        array: 2d numpy array
            Data to write
        path: str
            Path to the folder with source GeoTiff file

    Returns:
    --------
        .tif file
    """
    # Random source file to copy geodata for creating GeoTiff
    fn = os.listdir(os.path.join(path, "Crop_Eurasia/Climate_future", ""))[-1]

    template = rasterio.open(os.path.join(path, "Crop_Eurasia/Climate_future", fn))
    dtype = array.dtype
    # set data type to save
    if dtype == "byte":
        rasterio_dtype = rasterio.unit8
    elif dtype == "float32":
        rasterio_dtype = rasterio.float32
    elif dtype == "float64":
        rasterio_dtype = rasterio.float64
    elif dtype == "int16":
        rasterio_dtype = rasterio.int16
    elif dtype == "int32":
        rasterio_dtype = rasterio.int32
    else:
        print("Not supported data type.")

    with rasterio.open(file_name, 'w', **template.meta) as dest:
        dest.write(array.astype(rasterio_dtype), 1)


def avg_future_pred(Climate_future, years_future,
                    LC_feature, Elv, LC,
                    clf, features):
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
            LC model
        clf: xgbc model
            Prefitted model
        features: list[str]
            Features of the model

    Returns:
    --------
        prediction_prob: [1d numpy array]
            Future predictions flattened    
    """
    X_future, y_test, h, w = collect_data(Climate_future['CNRM-CM5'],
                                          years_future,
                                          LC_feature, Elv,
                                          LC, features, verbose=False)
    prediction_prob = clf.predict_proba(X_future)[:, 1]

    return prediction_prob, y_test, h, w


def changes2raster(clf, model_name, features, Elv,
                   LC_feature, Climate, LC,
                   years, year_current, path, path_pics, thresh):
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
    prediction_prob, y_test, h, w = avg_future_pred(Climate, years,
                                                    LC_feature, Elv, LC,
                                                    clf, features)

    # See the difference between prediction and the baseline
    prediction = [1 if x >= thresh else 0 for x in prediction_prob]
    y_proba = prediction_prob - y_test
    y = prediction - y_test

    # Reshape to correlate with sourse rasters
    y_proba_rect = y_proba.reshape(h, w)
    y_rect = y.reshape(h, w)
    y_test = y_test.reshape(h, w)

    # Check if the folder for record exists
    folder = os.path.join(path_pics, model_name)
    if os.path.exists(folder) is False:
        os.makedirs(folder)

    # Save as tif
    year = str(year_current)
    
    array2raster(os.path.join(folder, f"proba_{year}.tif"),
                 prediction_prob.reshape(h, w), path)
    array2raster(os.path.join(folder, f"changes_proba_{year}.tif"),
                 y_proba_rect, path)
    array2raster(os.path.join(folder, f"changes_{year}.tif"),
                 y_rect, path)
    array2raster(os.path.join(path_pics, "basic.tif"),
                 y_test, path)


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
    fn = os.listdir(os.path.join(path, "Crop_Eurasia/Climate_future", ""))[-1]

    template = rasterio.open(os.path.join(path, "Crop_Eurasia/Climate_future", fn))

    # Crop the tif
    with rasterio.open(source_tif) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = template.meta

    out_meta.update({"driver": "GTiff",
                     "count": count,
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    # Write new tif
    with rasterio.open(file_name, 'w', **out_meta) as dest:
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
        with fiona.open(os.path.join(path, "boundary", f"gadm41_{prefix}_0.shx"), "r") as sf:
            shapes = [feature["geometry"] for feature in sf]

        # Crop the country pic with classes existed in historical year
        crop_raster(os.path.join(path_pics, f"basic_{country_name}.tif"),
                    os.path.join(path_pics, "basic.tif"),
                    1, shapes, path)

        # Check if the folder for record exists
        folder = os.path.join(path_pics, model_name)
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        # Crop the changes picture
        year = str(year)
        crop_raster(os.path.join(folder,
                                 f"changes_proba_{year}_{country_name}.tif"),
                    os.path.join(folder,
                                 f"changes_proba_{year}.tif"),
                    1, shapes, path)
        crop_raster(os.path.join(folder,
                                 f"changes_{year}_{country_name}.tif"),
                    os.path.join(folder,
                                 f"changes_{year}.tif"),
                    1, shapes, path)


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
        Yeild: Dict
            Dictionary with rolling mean of Y_train and Y_test
    """
    # Rolling mean window parameter
    window = kwargs.get('window', 2)

    plt.figure(figsize=(30, 12))
    Yield = dict.fromkeys(prefixes)

    # Country data for rice
    for prefix in prefixes:
        c = np.random.rand(3,)
        Y_prefix = Production[(Production[:, 2] == prefix) &
                              (Production[:, 4] == 'Rice')]
        # years available
        years_country = pd.Series(list(map(int, Y_prefix[:, 3])))
        # yield available
        yield_country = pd.Series(list(map(float, Y_prefix[:, 0])),
                                  index=years)
        yield_country = yield_country.rolling(window=window,
                                              center=True
                                              ).mean()
        plt.plot(years_country, yield_country, label=prefix, color=c)
        Yield[prefix] = yield_country
    plt.legend(bbox_to_anchor=(1.04, 0.83), loc='center left')
    plt.suptitle('Rice yield')

    return Yield


def fert_forecast(Fert, Total_area,
                  prefixes, country_names,
                  years, **kwargs):
    """Plots fertilezers forecast

    Parameters:
    --------
        Fert: Dict
            Fert data for year, type, country
        Total_area: Dict
            Total arable area data for year, country
        prefixes: List[str]
            Short country names
        country_names: List[str]
            Normal spelling of country
        years: List[int]
            Years listed in FAOSTAT data
    Returns:
    --------
        Fert_future: Dict
    """
    n_years = kwargs.get('n_years', 10)
    years_future_fert = np.arange(years[-1], years[-1]+n_years)
    # n_years = x#len(years_future_fert)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(60, 20))
    Fert_future = {keys: [] for keys in prefixes}

    for prefix, name in zip(prefixes, country_names):
        values_N, values_P205, values_K2O = [], [], []
        c = np.random.rand(3,)
        for year in years:
            values_N.append(float(Fert[prefix][year][0])/Total_area[prefix][year]['total'])
            values_P205.append(float(Fert[prefix][year][1])/Total_area[prefix][year]['total'])
            values_K2O.append(float(Fert[prefix][year][2])/Total_area[prefix][year]['total'])

        values_N_diff = values_N[-1] - values_N[-n_years]
        values_N_future = np.copy(values_N[-n_years:]) + values_N_diff
        ax[0].plot(years, values_N, label=name, color=c)
        ax[0].plot(years_future_fert, values_N_future,
                   marker='.', color=c, linestyle='--')
        Fert_future[prefix].append(values_N_future)

        values_P205_diff = values_P205[-1] - values_P205[-n_years]
        values_P2O5_future = np.copy(values_P205[-n_years:]) + values_P205_diff
        ax[1].plot(years, values_P205, label=name, color=c)
        ax[1].plot(years_future_fert, values_P2O5_future,
                   marker='.', color=c, linestyle='--')
        Fert_future[prefix].append(values_P2O5_future)

        values_K2O_diff = values_K2O[-1] - values_K2O[-n_years]
        values_K2O_future = np.copy(values_K2O[-n_years:]) + values_K2O_diff
        ax[2].plot(years, values_K2O, label=name, color=c)
        ax[2].plot(years_future_fert, values_K2O_future,
                   marker='.', color=c, linestyle='--')
        Fert_future[prefix].append(values_K2O_future)

    ax[0].set_xlabel('Year')
    ax[1].set_xlabel('Year')
    ax[2].set_xlabel('Year')
    ax[0].set_ylabel('kg/ha')
    ax[1].set_ylabel('kg/ha')
    ax[2].set_ylabel('kg/ha')
    ax[0].set_title('N')
    ax[1].set_title('P2O5')
    ax[2].set_title('K2O')
    plt.legend(bbox_to_anchor=(0.35, 0.75), ncol=2)

    return Fert_future

def collect_data_yield(path_faostat, data_production,
                       prefixes, country_names,
                       Climate, Fert, Yield,
                       Total_area, crops, vars,
                       years, n_feat):
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

    reader = csv.reader(open(os.path.join(path_faostat, data_production), 'r'))

    for data in reader:
        country_prod = data[0]
        crop_prod = data[1]
        year_prod = int(data[2])

        const_0 = Trend.copy()

        for prefix, country_name in zip(prefixes, country_names):
            if (country_name in country_prod) & (year_prod <= years[-1]) & (year_prod >= years[1]):
                climate = Climate[prefix][year_prod]
                climate_array = np.stack([climate[var] for var in vars],
                                         axis=1).reshape(1, -1, order='F')

                fert_array = [x/Total_area[prefix][year_prod]['total'] for x in Fert[prefix][year_prod]]
                crop_area = Total_area[prefix][year_prod]['Rice']
                time_trend = year_prod - years[0]

                const_0 = Trend.copy()
                const_0[prefix] = time_trend
                const_array_0 = np.stack([const_0[prefix] for prefix in prefixes],
                                         axis=-1).reshape(1, -1)

                const_1 = Const_country.copy()
                const_1[prefix] = 1
                const_array_1 = np.stack([const_1[prefix] for prefix in prefixes],
                                         axis=-1).reshape(1, -1)

                const_2 = Const_crop.copy()
                const_2[crop_prod] = 1
                const_array_2 = np.stack([const_2[crop] for crop in crops],
                                         axis=-1).reshape(1, -1)

                array = np.hstack((
                                    climate_array[0],
                                    fert_array,
                                    const_array_0[0],
                                    const_array_1[0],
                                    const_array_2[0])
                                    ).astype(float)
                if year_prod <= years[-3]:
                    Y_train = np.vstack((Y_train, [Yield[prefix][year_prod],
                                                   Yield[prefix][year_prod]*crop_area,
                                                   prefix, year_prod, crop_prod]))
                    X_train = np.vstack((X_train, array))

                # 2 last years for test
                elif year_prod in [years[-2], years[-1]]:
                    Y_test = np.vstack((Y_test, [Yield[prefix][year_prod],
                                                 Yield[prefix][year_prod]*crop_area,
                                                 prefix, year_prod, crop_prod]))
                    X_test = np.vstack((X_test, array))

    return X_train, Y_train, X_test, Y_test


def collect_future_data(path_faostat, data_production,
                        prefixes, country_names,
                        Climate, Fert_future,
                        crops, vars,
                        years, years_future, n_feat):
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
    X_future = np.zeros((0, n_feat))

    # Empty dictionaries for including in feature space
    Const_country = {keys: 0 for keys in prefixes}
    Const_crop = {keys: 0 for keys in crops}
    Trend = {keys: 0 for keys in prefixes}

    reader = csv.reader(open(os.path.join(path_faostat, data_production), 'r'))
    for data in reader:

        if int(data[2]) == years[-1]:
            country_prod = data[0]
            crop_prod = data[1]

            for prefix, country_name in zip(prefixes, country_names):
                if (country_name in country_prod):

                    climate = Climate[prefix][years_future[0]]
                    climate_array = np.stack([climate[var] for var in vars],
                                             axis=1).reshape(1, -1, order='F')
                    fert_index = years_future[0] - years[-1]
                    fert_array = [Fert_future[prefix][0][fert_index],
                                  Fert_future[prefix][1][fert_index],
                                  Fert_future[prefix][1][fert_index]]

                    time_trend = years_future[0] - years[0]
                    const_0 = Trend.copy()
                    const_0[prefix] = time_trend
                    const_array_0 = np.stack([const_0[prefix] for prefix in prefixes],
                                             axis=-1).reshape(1, -1)

                    const_1 = Const_country.copy()
                    const_1[prefix] = 1
                    const_array_1 = np.stack([const_1[prefix] for prefix in prefixes],
                                             axis=-1).reshape(1, -1)

                    const_2 = Const_crop.copy()
                    const_2[crop_prod] = 1
                    const_array_2 = np.stack([const_2[crop] for crop in crops],
                                             axis=-1).reshape(1, -1)

                    array = np.hstack((climate_array[0],
                                      fert_array,
                                      const_array_0[0],
                                      const_array_1[0],
                                      const_array_2[0])
                                      ).astype(float)
                    X_future = np.vstack((X_future, array))
    return X_future


def yield_outcome(xgbr, Y_test, X_future_rice,
                  Total_area, neg_trend,
                  prefixes, year_base):
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
    year_base : int
        Year of the base year
    """
    columns_df = ['Country', 'Area_reduction',
                  'Yield_hist', 'Yield_future', 'Yield_ratio',
                  'Prod_hist', 'Prod_future', 'Prod_ratio']
    df = pd.DataFrame(columns=columns_df)
    n_count = len(prefixes)

    for i, prefix in enumerate(prefixes):
        df.loc[i, 'Area_reduction'] = - neg_trend[prefix]

        # yield historical
        conditions = (Y_test[:, 3] == str(year_base)) & (Y_test[:, 2] == prefix) & (Y_test[:, 4] == 'Rice')
        prod_hist = int(float(Y_test[conditions][0, 1]))
        df.loc[i, 'Country'] = prefix
        df.loc[i, 'Yield_hist'] = np.round(float(Y_test[conditions][0,0]), 2)
        df.loc[i, 'Prod_hist'] = prod_hist

        # crop production in future
        X_future_rice_local = X_future_rice[X_future_rice[:, -n_count+i] == 1]
        yield_future = xgbr.predict(X_future_rice_local)
        pred_future = yield_future*Total_area[prefix][year_base]['Rice']*(1-neg_trend[prefix]/100)
        df.loc[i, 'Prod_future'] = int(pred_future[0])
        df.loc[i, 'Yield_future'] = np.round(yield_future[0],2)

    df['Prod_ratio'] = (df['Prod_future'] - df['Prod_hist']) / df['Prod_hist']*100
    df['Yield_ratio'] = (df['Yield_future'] - df['Yield_hist']) / df['Yield_hist']*100
    df = df.astype({"Yield_ratio": np.float32, "Prod_ratio": np.float32})

    df.Prod_ratio = df['Prod_ratio'].round(decimals=1)
    df.Yield_ratio = df['Yield_ratio'].round(decimals=2)
    return df
