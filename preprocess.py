import sys
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig
sys.path.append(os.getcwd())
from src import load_data


@hydra.main(version_base=None,config_path=os.path.join(os.getcwd(), "configs"),config_name="main_config.yaml")
def run_prepare(cfg: DictConfig) -> None:
    path_train = cfg.process.path_raw_train
    path_predict= cfg.process.path_raw_predict
    path_processed = cfg.process.path_processed
    os.makedirs(path_processed, exist_ok=True)

    # If stop_year is not defined, start year goes to collection only
    year_start_train = cfg.year_start_train
    year_stop_train = cfg.year_stop_train

    year_start_test = cfg.year_start_test
    year_start_future = cfg.year_start_future
    
    # Land cover time lag (years)
    lag = cfg.process.lag
    LC_feature_lag = cfg.process.LC_feature_lag
    
    # List of CMIP simulations used
    CMIPs = cfg.train.CMIPs

    # Elevation
    Elv = load_data.elevation(path_train)
    with open(os.path.join(path_processed, 'elv_train.pkl'), 'wb') as f:
        pickle.dump(Elv, f)
    Elv = load_data.elevation(path_predict)
    with open(os.path.join(path_processed, 'elv_predict.pkl'), 'wb') as f:
        pickle.dump(Elv, f)

    # Land cover historical feature
    LC_feature_train = load_data.land_cover(path=path_train,
                                            year_start=year_start_train-LC_feature_lag,
                                            year_stop=year_stop_train-LC_feature_lag)
    LC_feature_test = load_data.land_cover(path=path_train,
                                            year_start=year_start_test-LC_feature_lag)
    LC_feature_future = load_data.land_cover(path=path_predict,
                                            year_start=year_start_future-LC_feature_lag)
    LC_base_for_future = load_data.land_cover(path=path_predict,
                                            year_start=year_start_test-LC_feature_lag)
    
    with open(os.path.join(path_processed, 'LC_feature_train.pkl'), 'wb') as f:
        pickle.dump(LC_feature_train, f)
    with open(os.path.join(path_processed,  'LC_feature_test.pkl'), 'wb') as f:
        pickle.dump(LC_feature_test, f)
    with open(os.path.join(path_processed, 'LC_feature_future.pkl'), 'wb') as f:
        pickle.dump(LC_feature_future, f)
    with open(os.path.join(path_processed, 'LC_base_for_future.pkl'), 'wb') as f:
        pickle.dump(LC_base_for_future, f)

    # Climate
    Climate_train, years_train = load_data.climate(path=path_train,
                                                    year_start=year_start_train,
                                                    year_stop=year_stop_train)
    Climate_test, years_test = load_data.climate(path=path_train,
                                                year_start=year_start_test)

    # Climate future
    Climate_future = dict.fromkeys(CMIPs)
    for CMIP in CMIPs:
        Climate_future[CMIP], years_future = load_data.climate(path=path_predict,
                                                                year_start=year_start_future,
                                                                CMIP=CMIP)
    with open(os.path.join(path_processed, 'Climate_train.pkl'), 'wb') as f:
        pickle.dump(Climate_train, f)
    with open(os.path.join(path_processed, 'Climate_test.pkl'), 'wb') as f:
        pickle.dump(Climate_test, f)
    with open(os.path.join(path_processed, 'Climate_future.pkl'), 'wb') as f:
        pickle.dump(Climate_future, f)

    # Land cover target feature
    LC_train = load_data.land_cover(path=path_train,
                                    year_start=year_start_train + lag,
                                    year_stop=year_stop_train + lag)
    LC_test = load_data.land_cover(path=path_train,
                                    year_start=year_start_test + lag)
    with open(os.path.join(path_processed, 'LC_train.pkl'), 'wb') as f:
        pickle.dump(LC_train, f)
    with open(os.path.join(path_processed, 'LC_test.pkl'), 'wb') as f:
        pickle.dump(LC_test, f)

    years = {}
    years['years_train'] = years_train
    years['years_test'] = years_test
    years['years_future'] = years_future
    with open(os.path.join(path_processed, 'years.pkl'), 'wb') as f:
        pickle.dump(years, f)


if __name__ == "__main__":
    run_prepare()