## Dependencies

* Dependencies are listed in environment.yml file.
* ex. Windows 11

## Data

The data was downloaded, preprocessed for the task and stored in .tif files. 
It is available at [Google Drive Folder](https://drive.google.com/drive/folders/1reYmmjR6ckznwakdeLyAC6DVKp3Adp2y?usp=sharing). This folder is referred to as `path` variable in notebooks.
Pay attention the available years range in that source:
* climate data is available from 1995 to 2020,
* future climate data is available for year 2025 with 3 different CMIP5 simulations,
* land cover data is available from 2001 to 2020,

## Executing program

[notebooks](https://github.com/DariTan/ArableLand/blob/master/notebooks) folder contains 3 independent scripts to run step by step.
Every notebook has a cell with years used in modelling and testing
1. [notebook_crop.ipynb](https://github.com/DariaTan/ArableLand/blob/master/notebooks/notebook_crop.ipynb)
* loads data
* performs some data analysis
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


[src](https://github.com/DariTan/ArableLand/blob/master/src) folder contains diverse .py modules for loading data, plotting and auxilary procedures
