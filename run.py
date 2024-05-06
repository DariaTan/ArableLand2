import sys,os

sys.path.append(os.getcwd())
from src import plotting, utils
from src.models.crop_classifier import classifier

from xgboost import XGBClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="main_config.yaml")
def run_inference(cfg:DictConfig):
    path_raw = cfg.process.path_raw
    path_processed = os.path.join(os.getcwd(), cfg.process.path_processed)
    path_pics = os.path.join(os.getcwd(), cfg.process.path_pics)
    if not os.path.exists(path_pics):
        os.makedirs(path_pics)

    model_path = os.path.join(cfg.eval.model_pickle,
                              f"{cfg.model_name}.pkl")
    params_xgbс = cfg.train.params_xgbс
    
    # Load model if it exist
    try: 
        model_loaded = XGBClassifier()
        with open(model_path,"rb") as fp:
            model_loaded = pickle.load(fp)
        print("model loaded, path: ", model_path)
        model_shell = classifier(model_loaded)
    except:
        print("First train the model")
        exit()

    # Define set of features to use in model
    set_features = cfg.train.set
    model_name = cfg.model_name
    
    if model_name == "model_climate_lc":
        set_features.append("lc")
    with open(os.path.join(path_processed, 'elv_train.pkl'), "rb") as f:
        Elv_train = pickle.load(f)
    with open(os.path.join(path_processed, 'elv_predict.pkl'), "rb") as f:
        Elv_pred = pickle.load(f)

    with open(os.path.join(path_processed, 'LC_feature_train.pkl'), "rb") as f:
        LC_feature_train = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_feature_test.pkl'), "rb") as f:
        LC_feature_test = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_feature_future.pkl'), "rb") as f:
        LC_feature_future = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_base_for_future.pkl'), "rb") as f:
        LC_base_for_future = pickle.load(f)

    with open(os.path.join(path_processed, 'Climate_train.pkl'), "rb") as f:
        Climate_train = pickle.load(f)
    with open(os.path.join(path_processed, 'Climate_test.pkl'), "rb") as f:
        Climate_test = pickle.load(f)
    with open(os.path.join(path_processed, 'Climate_future.pkl'), "rb") as f:
        Climate_future = pickle.load(f)

    with open(os.path.join(path_processed, 'years.pkl'), "rb") as f:
        years = pickle.load(f)
    years_train = years['years_train']
    years_test = years['years_test']
    years_future = years['years_future']

    with open(os.path.join(path_processed, 'LC_train.pkl'), "rb") as f:
        LC_train = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_test.pkl'), "rb") as f:
        LC_test = pickle.load(f)
    
    model_climate = model_shell.model(**params_xgbс) 
    model_shell.collect_data(Climate_train,
                            Climate_test,
                            Elv_train,
                            LC_feature_train,
                            LC_feature_test,
                            LC_train,
                            LC_test,
                            years_train,
                            years_test,
                            set_features)
    model_scores = model_shell.scores()
    
    # Misclassified areas on the map
    _, _, _, _ = utils.avg_future_pred(Climate_future,
                                        years_future,
                                        LC_feature_future,
                                        Elv_pred, LC_feature_future,
                                        model_climate,
                                        set_features)

    # Save all changes to raster
    utils.changes2raster(model_climate,
                            cfg.model_name,
                            set_features,
                            Elv_pred,
                            LC_feature_future,
                            Climate_future,
                            LC_base_for_future,
                            years_test,
                            cfg.year_start_future+cfg.process.lag,
                            path_raw,
                            path_pics,
                            model_scores['threshold'])

    # Create cropped tiffs for every country in the list
    prefixes = cfg.eval.prefixes
    country_names = cfg.eval.country_names
    utils.crop_country(cfg.model_name,
                        prefixes,
                        country_names,
                        path_raw,
                        path_pics,
                        cfg.year_start_future+cfg.process.lag)
    
    print(cfg.year_start_future+cfg.process.lag)
    plotting.plot_trend(cfg.model_name, prefixes,
                        country_names,
                        cfg.year_start_future+cfg.process.lag,
                        cfg.process.path_raw,
                        cfg.process.path_pics)

if __name__ == "__main__":
    run_inference()
