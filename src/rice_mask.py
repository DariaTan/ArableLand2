import os
import rasterio.merge
from rasterio.plot import show
from rasterio.mask import mask
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

from shapely import geometry
import fiona

import numpy as np
from descartes import PolygonPatch

from utils import (
    reshape_tiffs,
    reshape_tiff,
    crop_tiff,
    proba_thresholding,
    crop_country_file,
)


# Working path
path_raw = "/app/data/00-raw/Rice_map/2019/Raw/"

# Define climate data source folder
path_new = "/app/data/00-raw/Rice_map/2019/Rough/"

# Define climate data source folder
path = "/app/data/00-raw/"

path_pics = "/app/data/03-results/"


# ### Worsen paddy rice map resolution (initial value is 10m)


# reshape_tiffs(path_raw, path_new, 1000, 1000)


# ### Merge rice .tiffs to single file


files = os.listdir(path_new)
src_files_to_mosaic = []

for fp in files:
    src = rasterio.open(os.path.join(path_new, fp))
    src_files_to_mosaic.append(src)


# Sew all rice files
mosaic, out_trans = rasterio.merge.merge(src_files_to_mosaic)

# Copy the metadata
out_meta = src.meta.copy()

# Update the metadata
out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        #   "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
    }
)

with rasterio.open(os.path.join(path, "Rice_map", "out.tif"), "w", **out_meta) as dest:
    dest.write(mosaic)

rasterio.open(os.path.join(path, "Rice_map", "out.tif")).bounds


# Plot rice mask with water mask
rice_file = os.path.join(path, "Rice_map", "out.tif")
rice_map = rasterio.open(rice_file)

water = rasterio.open(os.path.join(path, "Asia", "water_mask.tif"))


fig, ax = plt.subplots(figsize=(60, 25))
rasterio.plot.show(water, ax=ax, cmap="Blues")
rasterio.plot.show(rice_map, ax=ax, alpha=0.5, cmap="Greens")
plt.savefig("/app/data/03-results/rice_mask.png")


# ### Align rice map and crop model output (in bounds and resolution)


class crop_features:
    """Class holding features for cropping and reshape

    Returns:
    --------
        new object of the class
    """

    def __init__(self, path_border, path_grid):
        bounds = rasterio.open(path_border).bounds

        # Define coordinate bounds
        left = bounds.left
        top = bounds.top
        right = bounds.right
        bottom = bounds.bottom

        bbox = [left, bottom, right, top]

        res_x = rasterio.open(path_grid).transform[0]
        res_y = -rasterio.open(path_grid).transform[4]

        self.width = (right - left) / res_x
        self.height = (top - bottom) / res_y
        bbox_size = (self.height, self.width)

        self.transform = rasterio.transform.from_bounds(
            *bbox, width=bbox_size[1], height=bbox_size[0]
        )

        p1 = geometry.Point(left, bottom)
        p2 = geometry.Point(left, top)
        p3 = geometry.Point(right, top)
        p4 = geometry.Point(right, bottom)

        pointList = [p1, p2, p3, p4]
        self.poly = geometry.Polygon([i for i in pointList])


th_pos = 0.9
th_neg = 0.8
proba_thresholding(
    os.path.join(path_pics, "model_climate_lc", "changes_proba_2028.tif"),
    os.path.join(path_pics, "model_climate_lc", "changes_2028_th.tif"),
    th_neg=th_neg,
    th_pos=th_pos,
)
path_model = os.path.join(path_pics, "model_climate_lc", "changes_2028_th.tif")
file_step1 = rice_file[:-4] + "1.tif"
file_step2 = rice_file[:-4] + "2.tif"
file_step3 = rice_file[:-4] + "3.tif"

# New rice output (cropped)
bound = crop_features(path_model, rice_file)
crop_tiff(rice_file, file_step1, bound)

# New model output (cropped)
bound = crop_features(file_step1, file_step1)
crop_tiff(path_model, file_step2, bound)

# Reshape rice output
bound = crop_features(file_step1, file_step2)
reshape_tiff(file_step1, file_step3, bound)


# Plot all together
rice_map = rasterio.open(file_step3)
model_map = rasterio.open(path_model)
water = rasterio.open(os.path.join(path, "Asia", "water_mask.tif"))

fig, ax = plt.subplots(figsize=(50, 20))
rasterio.plot.show(water, ax=ax, cmap="Blues")
rasterio.plot.show(model_map, ax=ax, alpha=0.5, cmap="Reds")
rasterio.plot.show(rice_map, ax=ax, alpha=0.5, cmap="Greens")
plt.savefig("/app/data/03-results/changes_under_rice_mask.png")


# Cut few extra pixels
model = rasterio.open(file_step2).read(1)[1:-1, 1:-1]
rice = rasterio.open(file_step3).read(1)
rice.shape == model.shape


# Create new tif with model output multiplied by rice mask
# new_raster = rice * model
# adding those who transform from one to 1
## new field occurs iff there were at least 1 rice field in +- `maxpool_pixels`
## maxpool2d is applied to rice map, thus, making pixel without rice (value=0) the rice one (value=1). The value3 is temporarily transformed to 0
maxpool_pixels = 2
rice_maxpool = rice.copy()
rice_maxpool[np.nonzero(rice_maxpool == 3)] = 0
## here is the maxpool2d filtering
rice_maxpool = generic_filter(
    rice_maxpool,
    function=np.max,
    footprint=np.ones((maxpool_pixels, maxpool_pixels)),
    mode="reflect",
)
new_raster = rice.copy()
## apply additional condition that for current filter there were at leatst on rice pixel = 1 (maxpool result)
new_raster[np.nonzero((rice == 0) * (model == 1) * (rice_maxpool == 1))] = 1
new_raster[np.nonzero((rice == 1) * (model == -1))] = -1

with rasterio.open(file_step3) as src:
    profile = src.profile

with rasterio.open(
    os.path.join(path_pics, "rice_climate_lc", "changes_rice.tif"), "w", **profile
) as dst:
    # Read the data and write it to the output raster
    dst.write(new_raster, 1)

values, counts_2018 = np.unique(new_raster, return_counts=True)
print(values, counts_2018)


# pay attention, the same data from tif show different unique values with same frequency
# 3 and -3 are NaNs - checked original dataset. It has 0, 1 and -2147483648, looks like water
# -3 -> 253; -1 -> 255
new_raster = rasterio.open(
    os.path.join(path_pics, "rice_climate_lc", "changes_rice.tif")
).read(1)
values, counts_2018 = np.unique(new_raster, return_counts=True)
print(values, counts_2018)


# ### Count risks withing country borders


country_names = [
    "Bangladesh",
    "Cambodia",
    "Japan",
    "Indonesia",
    "Lao PDR",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Korea",
    "Thailand",
    "Viet Nam",
]
prefixes = ["BGD", "KHM", "JPN", "IDN", "LAO", "MYS", "MMR", "PHL", "KOR", "THA", "VNM"]


# Run this cell for --changes_rice file and --rice_2018 (replace IN crop_country function and run again)
crop_country_file(
    prefixes, country_names, path, path_pics, file_to_crop="rice_2018.tif"
)
crop_country_file(
    prefixes, country_names, path, path_pics, file_to_crop="changes_rice.tif"
)


positive_changes = dict.fromkeys(prefixes)  # crop lands appeared
overall_changes = dict.fromkeys(prefixes)  # crop lands appeared + disappeared
negative_changes = dict.fromkeys(prefixes)  # crop lands disappeared

for prefix, country_name in zip(prefixes, country_names):
    print(country_name)

    # Load administrative borders
    with fiona.open(
        os.path.join(path, "boundary", f"gadm41_{prefix}_0.shx"), "r"
    ) as sf:
        shapes = [feature["geometry"] for feature in sf]
    patches = [
        PolygonPatch(shape, edgecolor="black", facecolor="none", linewidth=1)
        for shape in shapes
    ]

    basic = rasterio.open(
        os.path.join(path_pics, "rice_climate_lc", f"rice_2018_{country_name}.tif")
    )
    # basic = rasterio.open(rice_file)
    values_2018, counts_2018 = np.unique(basic.read(), return_counts=True)
    print(values_2018, counts_2018)

    changes = rasterio.open(
        os.path.join(path_pics, "rice_climate_lc", f"changes_rice_{country_name}.tif")
    )
    values, counts = np.unique(changes.read(), return_counts=True)
    print(values, counts)

    # dissapeared: 1-->255
    if len(np.where(values == 255)[0]) != 0 and len(np.where(values_2018 == 1)[0]) != 0:
        idx = np.where(values == 255)[0][0]
        idx_1 = np.where(values == 1)[0][0]
        idx_2018 = np.where(values_2018 == 1)[0][0]
        negative_changes[prefix] = np.round(
            counts[idx] / counts_2018[idx_2018] * 100, 1
        )
        overall_changes[prefix] = -np.round(
            (counts[idx_1] - counts_2018[idx_2018]) / counts_2018[idx_2018] * 100, 1
        )
        positive_changes[prefix] = overall_changes[prefix] + negative_changes[prefix]
    else:
        negative_changes[prefix] = 0
        positive_changes[prefix] = 0
        overall_changes[prefix] = positive_changes[prefix] + negative_changes[prefix]
    # positive_changes[prefix] = np.round(counts[1]/counts_2018[1]*100, 1)


# pickle
with open(
    "/app/data/03-results/model_climate_lc/crops_negative_2021_2028.pickle", "wb"
) as handle:
    pickle.dump(negative_changes, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(
    "/app/data/03-results/model_climate_lc/crops_positive_2021_2028.pickle", "wb"
) as handle:
    pickle.dump(positive_changes, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(
    "/app/data/03-results/model_climate_lc/crops_overall_2021_2028.pickle", "wb"
) as handle:
    pickle.dump(overall_changes, handle, protocol=pickle.HIGHEST_PROTOCOL)
# '/app/data/03-results/model_climate_lc/crops_negative_2021_2028.pickle'
