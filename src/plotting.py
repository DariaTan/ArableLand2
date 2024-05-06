from tqdm import tqdm
import numpy as np
import pandas as pd
import shap
import os, sys

sys.path.append(os.getcwd())
from src import utils, load_data

# Geospatial
import rasterio
import rasterio.plot
import fiona

# Visualize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from descartes import PolygonPatch
import seaborn as sns
import pickle

from hydra import initialize, compose

# Default plotting parameters
font = {"size": 16}
matplotlib.rc("font", **font)

# Custom gradient map
my_gradient = LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        (0.000, (0.898, 0.000, 0.000)),
        (0.250, (0.965, 0.604, 0.604)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.750, (0.627, 0.961, 0.639)),
        (1.000, (0.165, 0.494, 0.098)),
    ),
)

with initialize(version_base=None, config_path="../configs"):
    cfg = compose(config_name="main_config.yaml")

# # Shape of source raster
# Elv = load_data.elevation(cfg.process.path_raw)
# h = Elv.shape[0]
# w = Elv.shape[1]


def plot_biovars_dist(dict1, dict2):
    """Plots distribution of biovariables comparing 2 climate datasets
       (1 year from each one)

    Parameters:
    --------
        dict1: Dict
            Dictionary with biovariables
        dict2: Dict
            Dictionary with biovariables

    Returns:
    --------
        distribution plots
    """
    bio = [
        "bio1",
        "bio2",
        "bio3",
        "bio4",
        "bio5",
        "bio6",
        "bio7",
        "bio12",
        "bio13",
        "bio14",
        "bio15",
    ]
    biovars_names = [
        "Annual Mean Temperature",
        "Mean Diurnal Range",
        "Isothermality",
        "Temperature std",
        "Max Temperature",
        "Min Temperature",
        "Temperature Annual Range",
        "Annual Precipitation",
        "Precipitation of Wettest Period",
        "Precipitation of Driest Period",
        "Precipitation Seasonality",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(25, 12))
    axes = axes.ravel()

    n = dict1[bio[0]].shape[0]  # pixels in 1 raster

    # Loop over biovariables
    for i, (var, name) in tqdm(enumerate(zip(bio, biovars_names))):
        data = dict1[var][:n].ravel()  # Takes first year only
        sns.histplot(
            data, color="y", ax=axes[i], legend="test year", alpha=0.5, bins=100
        )

        data_future = dict2[var][:n].ravel()  # Takes first year only
        sns.histplot(
            data_future,
            color="b",
            ax=axes[i],
            legend="future year",
            alpha=0.5,
            bins=100,
        )

        axes[i].title.set_text(name)

    # Hide the frame and axis of the last subplot and put the legend there
    axes[11].set_frame_on(False)
    axes[11].get_xaxis().set_visible(False)
    axes[11].get_yaxis().set_visible(False)
    patch1 = mpatches.Patch(color="y", label="first dataset")
    patch2 = mpatches.Patch(color="b", label="second dataset")
    plt.legend(handles=[patch1, patch2])

    fig.suptitle("Climate biovariables distributions")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.show()


def plot_biovars_dist_changed(Climate_train, LC_train, LC_test):
    """Plots distribution of biovariables among objects changed their class
    Biovariables are taken from 2 climate datasets (1 year from each one)

    Parameters:
    --------
        Climate_train: Dict
            Train climate data
        Climate_test: Dict
            Test climate data
        LC_train: List
            Train land cover target feature
        LC_test: List
            Test land cover target feature

    Returns:
    --------
        distribution plots
    """
    biovars = [
        "bio1",
        "bio2",
        #    'bio3',
        # 'bio4',
        "bio5",
        "bio6",
        "bio7",
        "bio12",
        #    'bio13',
        #  'bio14',
        #  'bio15'
    ]
    biovars_names = [
        "Annual Mean Temperature",
        "Mean Diurnal Range",
        #  'Isothermality',
        #  'Temperature std',
        "Max Temperature",
        "Min Temperature",
        "Temperature Annual Range",
        "Annual Precipitation",
        #  'Precipitation of Wettest Period',
        #  'Precipitation of Driest Period',
        #  'Precipitation Seasonality'
    ]

    Climate_train_reshaped = {keys: [] for keys in biovars}

    # Take only first year from climate data
    for var in biovars:
        Climate_train_reshaped[var] = Climate_train[var][:, :, 0].reshape(-1)

    # Convert to pandas dataframe
    Biovars_train = pd.DataFrame.from_dict(
        Climate_train_reshaped, orient="columns", dtype=None, columns=None
    )

    # Dataframe with classes
    LC_pd = pd.DataFrame()
    LC_pd["train"] = LC_train[list(LC_train.keys())[0]].reshape(-1)
    LC_pd["test"] = LC_test[list(LC_test.keys())[0]].reshape(-1)

    # See the differences between datasets and write it to a separate column
    LC_pd["difference"] = LC_pd["test"] - LC_pd["train"]

    # List the indices of object whose class have been changed
    indices_to_crop_invert = list(np.where(LC_pd["difference"] != 1)[0])
    indices_to_crop = list(np.where(LC_pd["difference"] == 1)[0])
    indices_to_free_invert = list(np.where(LC_pd["difference"] != -1)[0])
    indices_to_free = list(np.where(LC_pd["difference"] == -1)[0])
    indices = set(indices_to_crop_invert).intersection(indices_to_free_invert)
    Biovars_train = Biovars_train.drop(indices, axis=0)

    # Set value in 'lc' column equal to 1/-1 for objects with changes
    Biovars_train.loc[indices_to_crop, "lc"] = 1
    Biovars_train.loc[indices_to_free, "lc"] = -1

    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(28, 10))
    axes = axes.ravel()
    # sns.set_style("whitegrid")

    # Loop over biovariables
    for i, (var, name) in tqdm(enumerate(zip(biovars, biovars_names))):
        sns.histplot(
            Biovars_train[var][Biovars_train["lc"] == 1],
            label="0--1",
            color="g",
            ax=axes[i],
            alpha=0.7,
            stat="density",
            bins=100,
        )
        sns.histplot(
            Biovars_train[var][Biovars_train["lc"] == -1],
            label="1--0",
            color="r",
            ax=axes[i],
            alpha=0.6,
            stat="density",
            bins=100,
        )

        axes[i].set_ylabel("Density", fontsize=28)
        axes[i].set_xlabel("{}, {}".format(name, var), fontsize=28)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig("dist.jpg", bbox_inches="tight", dpi=300)


def plot_trend(
    model_name,
    prefixes,
    country_names,
    year,
    path,
    path_pics,
    plot=False,
    plot_big=False,
) -> None:
    """Calculates changes in percents and plots them on map

    Parameters:
    --------
        model_name: str
            Name of the model
        prefixes: List[str]
            Short country names used in shapefiles
        country_name: List[str]
            Normal spelling of country
        year: int
            Year used
        path: str
            Path to data folder
        path_pics: str
            Path to save the new pic

    Returns:
    --------
        positive_changes: Dict
            The potential in percents (relative to historical crop lands)
        negative_changes: Dict
            The risk in percents (relative to historical crop lands)
    """
    # positive_changes = {key:0 for key in prefixes}  # crop lands appeared
    negative_changes = dict.fromkeys(prefixes)  # crop lands disappeared
    crop_initial = dict.fromkeys(prefixes)  # crop lands initial

    sns.reset_orig()
    shapes_all = []
    for prefix, country_name in zip(prefixes, country_names):
        # Load administrative borders
        with fiona.open(
            os.path.join(path, "boundary", f"gadm41_{prefix}_0.shx"), "r"
        ) as sf:
            shapes = [feature["geometry"] for feature in sf]
            shapes_all.extend(shapes)
        patches = [
            PolygonPatch(shape, edgecolor="black", facecolor="none", linewidth=1)
            for shape in shapes
        ]

        basic = rasterio.open(os.path.join(path_pics, f"basic_{country_name}.tif"))

        _, counts = np.unique(basic.read(), return_counts=True)
        crop_initial[prefix] = counts[1]
        year = str(year)

        raster_proba = rasterio.open(
            os.path.join(
                path_pics, model_name, f"changes_proba_{year}_{country_name}.tif"
            )
        )

        raster = rasterio.open(
            os.path.join(path_pics, model_name, f"changes_{year}_{country_name}.tif")
        )

        values, counts = np.unique(raster.read(), return_counts=True)
        # print(value, "counts", counts)
        if len(np.where(values == -1)[0]) != 0:
            idx = np.where(values == -1)[0][0]
            negative_changes[prefix] = np.round(
                counts[idx] / crop_initial[prefix] * 100, 1
            )
        # if len(np.where(value == 1)[0])!=0:
        #     idx= np.where(value == 1)[0][0]
        # positive_changes[prefix] = np.round(counts[2]/crop_initial[prefix]*100, 1)

        if plot:
            # Make plot for particular country
            fig, ax = plt.subplots(figsize=(4200 / 300, 2100 / 300), dpi=300)
            im = ax.imshow(raster_proba.read(1), cmap=my_gradient, vmin=-1, vmax=1)

            # Put raster and country border on the same plot
            rasterio.plot.show(raster_proba, ax=ax, cmap=my_gradient, vmin=-1, vmax=1)
            ax.add_collection(
                matplotlib.collections.PatchCollection(patches, match_original=True)
            )

            # Tune labels and colorbar
            # ax.set_xlabel('Longitude')
            # ax.set_ylabel('Latitude')
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            # cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            # cbar = fig.colorbar(im, cax=cax)
            # cbar.ax.set_yticklabels(['Risk', '', '', '', '', '', '', '', 'Potential'], fontsize=26)
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1, rect=[0, 0, 1, 1])
            plt.savefig(
                os.path.join(
                    path_pics, model_name, f"heatmap_{country_name}_{year}.jpg"
                ),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.show()

    if plot_big:
        # Make plot for all investigated area
        patches_all = [
            PolygonPatch(shape, edgecolor="black", facecolor="none", linewidth=1)
            for shape in shapes_all
        ]
        raster_proba_all = rasterio.open(
            os.path.join(path_pics, model_name, f"changes_proba_{year}.tif")
        )
        fig, ax = plt.subplots(figsize=(4200 / 300, 2100 / 300), dpi=300)
        im = ax.imshow(raster_proba_all.read(1), cmap=my_gradient, vmin=-1, vmax=1)

        # Put raster and country border on the same plot
        rasterio.plot.show(raster_proba_all, ax=ax, cmap=my_gradient, vmin=-1, vmax=1)
        ax.add_collection(
            matplotlib.collections.PatchCollection(patches_all, match_original=True)
        )

        # Tune labels and colorbar
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # ax.set_xlim(55, 118)
        # ax.set_ylim(0, 40)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=16)
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=16)
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_yticklabels(
            ["Risk", "", "", "", "", "", "", "", "Potential"], fontsize=26
        )
        plt.savefig(os.path.join(path_pics, model_name, f"heatmap_{year}.jpg"))
        plt.show()

    if not plot and not plot_big:
        with open(
            os.path.join(
                cfg.process.path_pics,
                cfg.model_name,
                f"crops_negative_{cfg.year_start_future-cfg.process.LC_feature_lag}_{cfg.year_start_future+cfg.process.lag}.pickle",
            ),
            "wb",
        ) as handle:
            pickle.dump(negative_changes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    basic, raster_proba = None, None


def plot_feature_imp(model, features) -> None:
    """Creates feature importances plot in shares

    Parameters:
    --------
        model: XGBmodel
            Prefitted model
        features: List[str]
            List of features used in model

    Returns:
    --------
        feature importance plot
    """
    # Boost feature names to the model
    model.get_booster().feature_names = features

    # Rate and sort features by score ('weight' criterion by default)
    feature_rating = model.get_booster().get_score()
    feature_rating_sorted = {
        k: v
        for k, v in sorted(
            feature_rating.items(), key=lambda item: item[1], reverse=True
        )
    }

    # Sum scores up
    total = sum(feature_rating_sorted.values())
    feat_names = list(feature_rating_sorted.keys())

    # Calculate share of every feature
    imp_dict = {
        feat_names[i]: feature_rating_sorted[feat_names[i]] / total
        for i in range(len(feat_names))
    }

    # Create horisonal bar plot
    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(imp_dict.keys()))

    sorted_result = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    ax.barh(y_pos, list(sorted_result[i][1] for i in range(len(sorted_result))), 0.2)

    # Plot top-to-bottom
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(sorted_result[i][0] for i in range(len(sorted_result))))
    ax.set_title("Feature importance of the model with features {}".format(features))
    plt.show()


def plot_shap(model, Climate, Elv, LC_feature, LC, years, features):
    """Creates feature importances plot

    Parameters:
    --------
        model: XGBmodel
            Prefitted model
        Climate_train: Dict
            Climate data
        Elv: List
            Elevation data
        LC_feature: 2d numpy array
            Land cover historical feature
        features: List[str]
            List of features used in model

    Returns:
    --------
        shap plot
    """
    # Compose dataset for the model
    X, _, _, _ = utils.collect_data(
        Climate, years, LC_feature, Elv, LC, features, verbose=False
    )

    # Explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model, feature_names=features)
    shap_values = explainer(X)

    # visualize the prediction's explanation
    shap.plots.bar(shap_values, show=False)
    plt.savefig("img.jpg", bbox_inches="tight", dpi=300)


def plot_changes(path_pics, model, year):
    """Plot changes to be expected in future comparing with test year

    Parameters:
    ----------
    path_pics: str
        Path to the folder with pictures
    model : str
        Name of the model
    year : int
        Year to consider

    Returns:
    -------
    plot
    """

    fig, ax = plt.subplots(figsize=(20, 10))

    basic = rasterio.open(
        os.path.join(path_pics, f"{model}", f"changes_proba_{year}.tif")
    )
    im_hidden = ax.imshow(basic.read(1), cmap=my_gradient, vmin=-1, vmax=1)
    rasterio.plot.show(
        basic.read(1),
        transform=basic.transform,
        ax=ax,
        cmap=my_gradient,
        vmin=-1,
        vmax=1,
    )

    # Tune labels and colorbar
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(im_hidden, cax=cax)
    cbar.ax.set_yticklabels(["Risk", "", "", "", "", "", "", "", "Potential"])
