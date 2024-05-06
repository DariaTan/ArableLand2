## Dependencies

* Dependencies are listed in 'environments/pyproject.toml' file.

## Data

The data was downloaded, preprocessed for the task and stored in .tif files. 
It is available at [Google Drive as archive](https://drive.google.com/file/d/1AIaQ07hfJPhwbenGdx3ZG7fZzNDumdcX/view?usp=sharing).
Prior to the running a docker container (see next section), the project directory should be organized as follows (tree depth is limited by 4):
``` bash
.
├── configs
│   ├── eval
│   |    └── eval_config.yaml
│   ├── process
│   |    └── process_config.yaml
│   ├── train
│   |    └── train_config.yaml
│   └── main_config.yaml
├── data
│   ├── 01-raw
│   |   ├── boundary
│   |   ├── Asia
│   |   |   ├── Climate
│   |   |   ├── LC_Type2.tif
│   |   |   └── ELv.tif
│   |   ├── Rice_map
│   |   ├── SEAsia
│   |   |   ├── Climate_future
│   |   |   ├── LC_Type2.tif
│   |   |   └── ELv.tif
│   |   └── FAOSTAT
│   ├── 01-prepared
│   ├── 02-models
│   ├── 03-results
│   └── 04-feed
├── README.md
├── environments
│   ├── Dockerfile
│   ├── poetry.lock
│   └── pyproject.toml
├── notebooks
│   ├── crops_statistics.ipynb
│   ├── notebook_yield.ipynb
│   ├── notebook_esg.ipynb
│   └── rice_mask.ipynb
└── src
    ├── load_data.py
    ├── models
    ├── plotting.py
    ├── rice_mask.py 
    └── utils.py
```

Pay attention the available years range in that source:
* climate data 'data/Asia/Climate' is available from 1966 to 2021,
* future climate data 'data/SEAsia/Climate_future' is available for years 2027, 2028 with 3 different CMIP5 simulations,
* land cover data is available from 2001 to 2022,

## Docker

From repo folder run:

* `docker build -t arable:1.0 environments/.`
* `docker run -it  -v  <CODE FOLDER>:/app -v <DATA FOLDER>:/app/data  -w="/app" arable:1.0`
* Inside the container run `sh download.sh` -- it will download all the necessary data from google drive, unpack it, delete the archived data

## Executing program

Scripts in repo folder allow to perform main steps of algorithm.
To run them properly, set the required parameters in all files (including subfolders) of 'config folder'.
* preprocess.py performs data processing and saves data in 01-prepared folder
* train.py trains the model
* run.py creates several plots

Alternatively, the same steps are implemented interactevely in notebook_esg.ipynb

Moreover, notebook_yield.ipynb performs modeling of fertilizers consumption and rice yield in countries of Southeastern Asia. More respecifically, it
* loads data
* defines set to countries to perform further analysis
* collects climate statistics for those countries
* builds SARIMAX model to predict fertilizers consumption for those countries
* performs yield modeling with XGBRegressor
* estimates the uncertainty of that model with boostrap procedure

'src' folder contains `load_data.py`, `plotting.py` and `utils.py` modules for loading data, plotting and auxilary procedures.