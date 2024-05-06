import sys,os
sys.path.append(os.getcwd())
from src.models.crop_classifier import classifier


import pickle
import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="main_config.yaml")
def train_model(cfg:DictConfig):

    path_processed = os.path.join(os.getcwd(), cfg.process.path_processed)
    path_pics = os.path.join(os.getcwd(), cfg.process.path_pics)
    if not os.path.exists(path_pics):
        os.makedirs(path_pics)

    model_path = os.path.join(cfg.eval.model_pickle,
                              f"{cfg.model_name}.pkl")
    params_xgbс = cfg.train.params_xgbс
    model_loaded = None
    print("Training model...")

    # Define set of features to use in model
    set_features = cfg.train.set
    model_name = cfg.model_name
    
    if model_name == "model_climate_lc":
        set_features.append("lc")
    with open(os.path.join(path_processed, 'elv_train.pkl'), "rb") as f:
        Elv = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_feature_train.pkl'), "rb") as f:
        LC_feature_train = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_feature_test.pkl'), "rb") as f:
        LC_feature_test = pickle.load(f)

    with open(os.path.join(path_processed, 'Climate_train.pkl'), "rb") as f:
        Climate_train = pickle.load(f)
    with open(os.path.join(path_processed, 'Climate_test.pkl'), "rb") as f:
        Climate_test = pickle.load(f)

    with open(os.path.join(path_processed, 'years.pkl'), "rb") as f:
        years = pickle.load(f)
    years_train = years['years_train']
    years_test = years['years_test']

    with open(os.path.join(path_processed, 'LC_train.pkl'), "rb") as f:
        LC_train = pickle.load(f)
    with open(os.path.join(path_processed, 'LC_test.pkl'), "rb") as f:
        LC_test = pickle.load(f)
    
    model_shell = classifier(model_loaded)
    model_climate = model_shell.model(**params_xgbс)
    model_shell.collect_data(Climate_train,
                            Climate_test,
                            Elv,
                            LC_feature_train,
                            LC_feature_test,
                            LC_train,
                            LC_test,
                            years_train,
                            years_test,
                            set_features)

    # Fit and save model. Set type="all" to train several basic classifiers
    model_shell.fit_classifier(model_path=model_path, type="XGB")
    

if __name__ == "__main__":
    train_model()