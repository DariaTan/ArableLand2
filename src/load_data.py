import os
import sys
import csv
import numpy as np
from tqdm import tqdm
import fiona
import rasterio
import glob
sys.path.append(os.getcwd())
from src import utils


def elevation(path):
    """Collects elevation features

    Parameters:
    --------
        path: str
            Path to folder with data

    Returns:
    --------
        Elv: 2d numpy array
            Elevation values of objects
    """
    data = rasterio.open(os.path.join(path, "Elv.tif"))
    Elv = data.read(1)
    data = None
    return Elv
    

def land_cover(path, year_start, **kwargs):
    """Composes land cover feature

    Parameters:
    --------
        path: str
            Path to folder that containts data
        year_start: int
            Starting year to use for collection
        year_stop: Optional[int]
            Last year to use for collection (exclusive)

    Returns:
    --------
        LC: Dict[int, 2d numpy array]
    """
    year_stop = kwargs.get("year_stop", year_start + 1)
    years_feature = np.arange(year_start, year_stop)
    LC = {keys: [] for keys in years_feature}
    data = rasterio.open(os.path.join(path, "LC_Type2.tif"))
    
    # Count bands
    n_bands = data.count
    bands = np.arange(1, n_bands + 1)
    data_years = [data.descriptions[band - 1][:4] for band in bands]

    if str(year_start) not in data_years or str(year_stop-1) not in data_years:
        print("No land cover data for years required. Modify configuration file")
    
    # Choose the required years (necessary bands) only
    for year in years_feature:
        for band in bands:
            if str(year) == data.descriptions[band - 1][:4]:
                raster = data.read(int(band))
                # Switch to binary classification
                raster = np.where(raster == 12, 1, 0)  # LC_type = 12 is cropland
                LC[year] = raster

    data = None
    print(
        f"Land cover dictionary is collected for {len(years_feature)} year(s)"
    )
    return LC


def land_cover_multi(path, year_start, **kwargs):
    """Composes land cover feature

    Parameters:
    --------
        path: str
            Path to folder that containts data
        year_start: int
            Starting year to use for collection
        year_stop: Optional[int]
            Last year to use for collection (exclusive)

    Returns:
    --------
        LC: Dict[int, 2d numpy array]
    """
    year_stop = kwargs.get("year_stop", year_start + 1)
    years_feature = np.arange(year_start, year_stop)
    LC = {keys: [] for keys in years_feature}
    data = rasterio.open(os.path.join(path, "LC_Type2.tif"))
    n_bands = data.count  # number of bands in tif
    bands = np.arange(1, n_bands + 1)
    for year in years_feature:
        # Choose the required years (necessary bands) only
        for band in bands:
            if str(year) == data.descriptions[band - 1][:4]:
                raster = data.read(int(band))
                LC[year] = raster
    data = None
    print(
        "Land cover dictionary is collected for {} year(s)".format(len(years_feature))
    )
    return LC


def climate(path, year_start, **kwargs):
    """Composes the dictionary with biovariables from climate data.

    Parameters:
    --------
        path: str
            Path to folder that containts historical data
        year_start: int
            Starting year to use for collection
        year_stop: Optional [int]
            Last year year to use for collection (exclusive)
        CMIP: Optional[str]
            Name of the CMIP dataset

    Returns:
    --------
        Climate: Dict[str, 2d numpy array]
        years: List[str]
    """
    year_stop = kwargs.get("year_stop", year_start + 1)
    monthes = np.arange(1, 13)
    years = np.arange(year_start, year_stop)
    CMIP = kwargs.get("CMIP", "CMIP5")

    scenario = CMIP + "_rcp45_"  # used for future climate
    flag_future = False

    # Create mask to detect sea areas
    water_raster = rasterio.open(os.path.join(path,
                                            "water_mask.tif"))
    water_mask = water_raster.read(1) == 1
    w = water_raster.width  # raster width
    h = water_raster.height  # raster height

    # empty array
    arr = np.empty((h, w, 0))
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
    Climate = {keys: np.empty((h, w, 0)) for keys in bio}

    # Create pattern to reach the tif by name:
    # Historical data
    if year_start < 2023:
        path_clim = os.path.join(path, "Climate")

    # Future data
    else:
        flag_future = True
        path_clim = os.path.join(path, "Climate_future")

    # Numbers for specific year
    for year in tqdm(years):
        bio1, bio2, bio4, bio5, bio6, bio12 = arr, arr, arr, arr, arr, arr

        # Numbers for specific month
        for month in monthes:
            if flag_future:
                ending = str(year) + "_" + str(month) + "_avg.tif"

                fn = scenario + "tmmx_" + ending
                tmmx = rasterio.open(os.path.join(path_clim, fn))
                raster_tmmx = tmmx.read(1)/10

                fn = scenario + "tmmn_" + ending
                tmmn = rasterio.open(os.path.join(path_clim, fn))
                raster_tmmn = tmmn.read(1)/10

                fn = scenario + "pr_" + ending
                pr = rasterio.open(os.path.join(path_clim, fn))
                raster_pr = pr.read(1)

            else:
                ending = str(year) + ".tif"

                fn = "tmmx_" + ending
                tmmx = rasterio.open(os.path.join(path_clim, fn))
                raster_tmmx = tmmx.read(int(month))/10

                fn = "tmmn_" + ending
                tmmn = rasterio.open(os.path.join(path_clim, fn))
                raster_tmmn = tmmn.read(int(month))/10

                fn = "pr_" + ending
                pr = rasterio.open(os.path.join(path_clim, fn))
                raster_pr = pr.read(int(month))

            # Apply water mask
            raster_tmmx[water_mask] = 0
            raster_tmmn[water_mask] = 0
            raster_pr[water_mask] = 0

            t_avg = (raster_tmmx + raster_tmmn) / 2

            # Bio1. Annual Mean Temperature
            bio1 = np.dstack((bio1, t_avg))

            # Bio2. Mean Diurnal Range(Mean(period max-min))
            bio2_m = raster_tmmx - raster_tmmn
            bio2 = np.dstack((bio2, bio2_m))

            # Bio4. Temperature Seasonality (standard deviation)
            bio4 = np.dstack((bio4, t_avg))

            # Bio5. Max Temperature of Warmest Period
            bio5 = np.dstack((bio5, raster_tmmx))

            # Bio6. Min Temperature of Coldest Period
            bio6 = np.dstack((bio6, raster_tmmn))

            # Bio7. Temperature Annual Range (Bio5 - Bio6)

            # Bio3. Isothermality (Bio2 / Bio7)

            # Bio12. Annual Precipitation
            bio12 = np.dstack((bio12, raster_pr))

            # Bio13. Precipitation of Wettest Period
            # Bio14. Precipitation of Driest Period

            # Bio15. Precipitation Seasonality(Coefficient of Variation)
            # the "1 +" is to avoid strange CVs for areas where mean rainfaill is < 1)

        # Accumulate 12 month numbers into that year
        Climate["bio1"] = np.dstack((Climate["bio1"], np.mean(bio1, axis=2)))
        Climate["bio2"] = np.dstack((Climate["bio2"], np.mean(bio2, axis=2)))
        Climate["bio4"] = np.dstack((Climate["bio4"], np.std(bio4, axis=2)))
        Climate["bio5"] = np.dstack((Climate["bio5"], np.max(bio5, axis=2)))
        Climate["bio6"] = np.dstack((Climate["bio6"], np.min(bio6, axis=2)))
        Climate["bio7"] = np.dstack(
            (
                Climate["bio7"],
                np.array(Climate["bio5"])[:, :, -1]
                - np.array(Climate["bio6"])[:, :, -1],
            )
        )
        Climate["bio12"] = np.dstack((Climate["bio12"], np.sum(bio12, axis=2)))
        Climate["bio13"] = np.dstack((Climate["bio13"], np.max(bio12, axis=2)))
        Climate["bio14"] = np.dstack((Climate["bio14"], np.min(bio12, axis=2)))
        Climate["bio3"] = np.dstack(
            (
                Climate["bio3"],
                Climate["bio2"][:, :, -1]
                * 100
                / (Climate["bio7"][:, :, -1] + np.ones((h, w)) * 1e-9),
            )
        )  # add tiny value to avoid division by zero
        Climate["bio15"] = np.dstack(
            (
                Climate["bio15"],
                (np.std(bio12, axis=2) / (1 + (Climate["bio12"][:, :, -1]) / 12)),
            )
        )

    # Close raster
    tmmx = None
    tmmn = None
    pr = None
    print(
        "Output dictionary with climate includes {} biovariables accumulated for {} year(s)".format(
            len(Climate), len(years)
        )
    )
    return Climate, years


def production(path, csv_filename, prefixes, country_names, total_area, years):
    """
    Loads crops production data from a FAOSTAT .csv file

    Parameters
    ----------
    path : str
        Path to the folder containing the .csv files
    csv_filename : str
        .csv file containing with data
    prefixes : list[str]
        List of country prefixes
    country_names : list[str]
        List of country names
    total_area : dict
        Dictionary containing the total area for each country
    years : list[int]
        List of years

    Returns
    -------
    production : dict
        Dictionary containing rice production for each country and year
    """
    production = np.zeros((0, 5))

    reader = csv.reader(open(os.path.join(path, csv_filename), 'r'))
    for data in reader:
        country_prod = data[0]
        crop_prod = data[2]
        year_prod = int(data[3])
        for prefix, country_name in zip(prefixes, country_names):
            if (country_name in country_prod) & (year_prod in years):
                crop_area = total_area[prefix][year_prod]['Rice']
                production = np.vstack((production,
                                        [float(data[5])/crop_area,
                                         float(data[5]), prefix,
                                         year_prod,
                                         crop_prod]))
    return production


def arable_area(
    path, csv_arable, csv_arable_crops, prefixes, country_names, years, crops
):
    """
    Loads arable area from FAOSTAT .csv file

    Parameters
    ----------
    path : str
        Path to the folder containing the .csv files
    csv_arable : str
        Path to the .csv file containing the arable area
    csv_arable_crops : str
        Path to the .csv file containing the arable area per crop
    prefixes : list
        List of country prefixes
    country_names : list
        List of country names
    years : list
        List of years
    crops : list
        List of crops

    Returns
    -------
    Total_area : dict
        Dictionary containing the total arable area for each country and crop
    """

    Total_area = {keys: {} for keys in prefixes}
    for prefix in prefixes:
        Total_area[prefix] = {keys: {} for keys in years}

    with open(os.path.join(path, csv_arable), "r") as file1:
        reader1 = csv.reader(file1)
        for data in reader1:
            if data[1] == "Cropland":
                try:
                    year_prod = int(data[2])
                    if years[0] <= year_prod <= years[-1]:
                        country_prod = data[0]
                        crop_prod = float(data[4])

                        for prefix, country_name in zip(prefixes, country_names):
                            if country_name in country_prod:
                                Total_area[prefix][year_prod]["total"] = crop_prod * 1000
                except ValueError:
                    pass  # Skip rows with non-integer year values

    with open(os.path.join(path, csv_arable_crops), "r") as file2:
        reader2 = csv.reader(file2)
        for data in reader2:
            try:
                year_prod = int(data[3])
                if years[0] <= year_prod <= years[-1]:
                    country_prod = data[0]
                    crop_type = data[2]
                    crop_prod = int(data[5])

                    for prefix, country_name in zip(prefixes, country_names):
                        if country_name in country_prod:
                            for crop in crops:
                                if crop in crop_type:
                                    Total_area[prefix][year_prod].setdefault(crop, 0)
                                    Total_area[prefix][year_prod][crop] += crop_prod
            except ValueError:
                pass  # Skip rows with non-integer year values

    return Total_area


def fertilizers(path, csv_fert, prefixes, country_names, years):
    """
    Loads fertilizers from FAOSTAT .csv file

    Parameters
    ----------
    path : str
        Path to the folder containing the .csv files
    csv_fert : str
        Path to the .csv file containing the fertilizers
    prefixes : list[str]
        List of country prefixes
    country_names : list[str]
        List of country names
    years : list[int]
        List of years

    Returns
    -------
    Fert : dict
        Dictionary containing the fertilizers for each country and year
    """
    Fert = {}
    for prefix in prefixes:
        Fert[prefix] = {}
        for year in years:
            Fert[prefix][year] = []

    with open(os.path.join(path, csv_fert), 'r', encoding='latin-1') as file:
        reader = csv.reader(file)
        for data in reader:
            if len(data) > 7:
                year_str = data[7]
                if year_str.isdigit():
                    year = int(year_str)
                    country_prod = data[2]
                    for prefix, country_name in zip(prefixes, country_names):
                        if (country_name in country_prod) and (year in years):
                            val = data[10]
                            Fert[prefix][year].append(float(val))

    return Fert

    # for prefix, country_name in zip(prefixes, country_names):
    #     for year in years:
    #         Fert[prefix][year] = []
    #         reader = csv.reader(open(os.path.join(path, csv_fert), "r"))
    #         for data in reader:
    #             if country_name in data[0]:
    #                 val = data[year - years[0] + 3]
    #                 Fert[prefix][year].append(float(val))
    # return Fert


def climate_for_yield(
    path, prefixes, country_names, vars, years, years_future, CMIPs
):
    """
    Create croped .tiff files for each country
    Calculates climate data statistics based on:
        TerraClimate for historical years
        CMIP5 model for future years

    Parameters
    ----------
    path : str
        Path to the folders containing the files
    prefixes : list[str]
        List of country prefixes
    country_names : list[str]
        List of country names
    vars : list[str]
        List of variables
    years : list[int]
        List of years
    years_future : list[int]
        List of years in the future
    Returns
    -------
    Climate : dict[country, dict]
        Dictionary containing the climate data
    """

    Climate = {keys: {} for keys in prefixes}
    path_climate = os.path.join(path, "00-raw", "Asia", "Climate", "")
    path_climate_future = os.path.join(path, "00-raw", "SEAsia", "Climate_future", "")
    path_econ = os.path.join(path, "01-prepared", "Economics", "")
    os.makedirs(path_econ, exist_ok=True)
    monthes = range(1,13)
    
    # Create single multiband climate for this year (stack of 12 months) ###############################Required once for year in future
    # for var in vars:
    #     for year in years_future:            
    #         for model in CMIPs:
    #             # Create single multiband climate for this year (stack of 12 months)
    #             files_from = glob.glob(f"{path_climate_future}/{model}_rcp45_{var}_{year}_*_avg.tif")   
    #             file_to = f"{path_climate_future}/{model}_rcp45_{var}_{year}_avg.tif"
    #             out_meta = rasterio.open(files_from[0]).meta
    #             out_meta.update(count = len(monthes))
                
    #             with rasterio.open(file_to, 'w', **out_meta) as dest:
    #                 for band_nr, src in enumerate(files_from, start=1):
    #                     data = rasterio.open(src).read(1)
    #                     dest.write(data, band_nr)
                        
    for prefix in prefixes:
        # Load administrative border
        with fiona.open(
            os.path.join(path, "00-raw", "boundary", f"gadm41_{prefix}_0.shx"), "r"
        ) as sf:
            shapes = [feature["geometry"] for feature in sf]

        # Historical data
        for year in years:
            Climate[prefix][year] = {keys: [] for keys in vars}

            for var in vars:
                file_to = f"{var}_{year}.tif"
                utils.crop_raster(
                    path_econ + prefix + "_" + file_to,
                    path_climate + file_to,
                    len(monthes),
                    shapes,
                    f"{path}/00-raw",
                )

                # Calculate variables over the country
                with rasterio.open(path_econ + prefix + "_" + file_to) as tif:
                    for band in np.arange(1, tif.count + 1):
                        if var == "pr":
                            arr = tif.read(int(band)).ravel().astype(np.float32) + 0.001
                        else:
                            arr = (tif.read(int(band)).ravel() / 10 + 273.15).astype(
                                np.float32
                            )
                        mask = arr != 0
                        av = np.ma.masked_where(~mask, arr).mean()
                        variance = np.var(arr)
                        Climate[prefix][year][var].append(av)
                        Climate[prefix][year][var].append(variance)
        # Future data
        for year in years_future:
            Climate[prefix][year] = {keys: [] for keys in CMIPs}
            for model in CMIPs:
                Climate[prefix][year][model] = {keys: [] for keys in vars}
                for var in vars:
                    
                    file_to = f"{path_climate_future}{model}_rcp45_{var}_{year}_avg.tif"
                    utils.crop_raster(
                        f"{path_econ}/{prefix}_{var}_{year}.tif",
                        file_to,
                        len(monthes),
                        shapes,
                        f"{path}/00-raw",
                    )

                    # Calculate average over the country
                    with rasterio.open(f"{path_econ}/{prefix}_{var}_{year}.tif") as tif:
                        for band in np.arange(1, tif.count + 1):
                            if var == "pr":
                                arr = tif.read(int(band)).ravel().astype(np.float32) + 0.001
                            else:
                                arr = (tif.read(int(band)).ravel() / 10 + 273.15).astype(
                                    np.float32
                                )
                            mask = arr != 0
                            av = np.ma.masked_where(~mask, arr).mean()
                            variance = np.var(arr)
                            Climate[prefix][year][model][var].append(av)
                            Climate[prefix][year][model][var].append(variance)
    return Climate

