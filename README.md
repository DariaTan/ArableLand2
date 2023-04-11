## Dependencies

* Dependencies are listed in environment.yml file.

## Data

The data was downloaded, preprocessed for the task and stored in .tif files. 
It is available at [Google Drive Folder](https://drive.google.com/drive/folders/1reYmmjR6ckznwakdeLyAC6DVKp3Adp2y?usp=sharing). This folder is referred to as `path` variable in notebooks. 
Prior to the running a docker container (see next section), the project directory should be organized as follows (tree depth is limited by 2):
``` bash
.
├── Dockerfile
├── Geo_data
│   ├── Crop_Eurasia
│   ├── FAOSTAT
│   └── boundary
├── README.md
├── environments
│   ├── environment.yml
│   └── requirements.txt
├── notebooks
│   ├── crops_disappear_2019_2026.pickle
│   ├── model_climate.pkl
│   ├── model_climate_lc.pkl
│   ├── notebook_crop.ipynb
│   └── notebook_yield.ipynb
└── src
    ├── load_data.py
    ├── models
    ├── plotting.py
    └── utils.py
```

Pay attention the available years range in that source:
* climate data is available from 1995 to 2020,
* future climate data is available for year 2025 with 3 different CMIP5 simulations,
* land cover data is available from 2001 to 2020,

## Docker

From repo folder run:

* `docker build -t crop_dev .`
* `docker run -it  -v  <CODE FOLDER>:/crop -v <DATA FOLDER>:/crop/Geo_data -m 16000m  --cpus=4  -w="/crop" crop_dev`
## Executing program

[notebooks](https://github.com/DariTan/ArableLand/blob/master/notebooks) folder contains 2 independent scripts to run step by step.
Every notebook has a cell with years used in modeling and testing
1. [notebook_crop.ipynb](https://github.com/DariaTan/ArableLand/blob/master/notebooks/notebook_crop.ipynb)
* loads data
* plots distributions of climate variables and distribution of climate features condition on class - aquired/lost crop status
* builds 2 models with slightly different set of features to predict the land status (binary classification task)
* evaluates those models on test data (with some classification metrics output)
* evaluates feature importances for both models
* plots type I and type II errors on the country maps for the test year
* counts the fraction of crop lands subject to disappear for each country
2. [notebook_yield.ipynb](https://github.com/DariTan/ArableLand/blob/master/notebooks/notebook_yield.ipynb) 
* loads data for some countries
* creates linear regression model for each country to predict fertilizers consumption
* builds a regression model to predict the yield of rice
* evaluates the model on test data
* calculates production in future year and its relative change comparing to baseline


[src](https://github.com/DariTan/ArableLand/blob/master/src) folder contains `load_data.py`, `plotting.py` and `utils.py` modules for loading data, plotting and auxilary procedures