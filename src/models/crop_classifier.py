from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import utils
import pickle

def model_grid_search(Climate_train, Elv, LC_feature_train, LC_train,
                      years_train, features):
    """Runs XGBClassification with all the features included
    to evaluate best parameters

    Parameters:
    --------
        Climate_train: Callable[Dict]
            Train climate data
        Elv: Callable[2d numpy array]
            Elevation data
        LC_feature_train: Callable[Dict]
            Land cover historical feature
        LC_train: Callable[Dict]
            Train land cover target feature
        years_train: Callable[List[int]]
            Years used for train
        features: Callable[List[str]]
            Feature names used in model

    Returns:
    --------
        xgbc - GridSearchCV object
    """
    params_xgb = {
        'learning_rate': 0.02,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'nthread': -1,
        'eval_metric': 'logloss'
        }
    xgb = XGBClassifier(**params_xgb)

    # Compose dataset for the model
    X_train, y_train = utils.collect_data(Climate_train, years_train,
                                          LC_feature_train, Elv, LC_train,
                                          features, verbose=True)

    # StratifiedKFold provides cross validations keeping shares of classes in folds
    params_skf = {
            'n_splits': 3,
            'shuffle': True
            }
    skf = StratifiedKFold(**params_skf, random_state=1001)

    # Set of parameters for Grid Search
    params_gs = {
                'max_depth': [3, 5],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 10, 100]
                }
    xgbc = GridSearchCV(xgb, param_grid=params_gs,
                        scoring='balanced_accuracy',
                        cv=skf)

    # Fit the model
    xgbc.fit(X_train, y_train)

    print('\n Best estimator:')
    print(xgbc.best_estimator_)
    print('\n Best hyperparameters:')
    print(xgbc.best_params_)
    print('\n Best balanced accuracy score')
    print(xgbc.best_score_)

    return xgbc


class classifier():
    """Class including XGBClassification model and some methods relating to it
    """
    def __init__(self, loaded_model=None):
        self.loaded_model = loaded_model

    def model(self, **kwargs):
        """Loads prefitted model if .pkl file exists
        Otherwise creates XGBClassifier model with params defined

        Parameters:
        --------
            **kwargs: [Dict]
                Parameters of XGBClassifier
        Returns:
        --------
            model (XGBClassifier)
        """
        if self.loaded_model is not None:
            self.xgbc_model = self.loaded_model
        else:
            self.xgbc_model = XGBClassifier(**kwargs)
        return self.xgbc_model

    def collect_data(self, Climate_train, Climate_test, Elv,
                     LC_feature_train, LC_feature_test, LC_train, LC_test,
                     years_train, years_test, features):
        """    Collects data for the model

        Parameters:
        --------
            Climate_train (dict): - climate train data
            Climate_test (dict): - climate test data
            Elv (2d array): - elevation data
            LC_feature_train (dict): - land cover historical feature, train
            LC_feature_test (dict): - land cover historical feature, test
            LC_train (dict): -land cover target feature, train data
            LC_test (dict): -land cover target feature, test data
            years_train (list of int): - years used in train
            years_test (list of int): - years used in test
            features (list of str): - feature names used in model

        Returns:
        --------
            X_train (2d array): - all the instances with attributes, train data
            y_train (1d array): - label of each instance, train data
            X_test (2d array): - all the instances with attributes, test data
            y_test (1d array): - label of each instance, test data
        """
        self.X_train, self.y_train, self.h, self.w = utils.collect_data(Climate_train, years_train,
                                                        LC_feature_train, Elv, LC_train,
                                                        features, verbose=False)
        self.X_test, self.y_test, self.h, self.w = utils.collect_data(Climate_test, years_test,
                                                      LC_feature_test, Elv, LC_test,
                                                      features, verbose=False)
        self.features = features
        return self.X_train, self.y_train, self.X_test, self.y_test, self.features, self.h, self.w

    def fit_classifier(self):
        """Fits the model if .pkl file doesn't exist

        Returns:
        --------
            self.xgbc_model.fit
        """
        if self.loaded_model is None:
            return self.xgbc_model.fit(self.X_train, self.y_train)

    def scores(self):
        """Calculates scores of the model on test data

        Returns:
        --------
        self.scores: Callable[Dict[metric, value]]
            Dictionary with model metrics
        """
        self.scores = utils.xgbc_scores_binary(self.xgbc_model, 
                                               self.X_test, self.y_test)
        print('Model with features', self.features)
        for k, v in self.scores.items():
            print(k, '=', v)
        return self.scores

    def save(self, filepath, rewrite=False):
        """
        Saves prefitted model to .pkl file

        Parameters:
        --------
            filepath: Callable[str]
                path to save file at
            rewrite: Optional[Bool]
                flag, if it is True, rewrites
        """
        if rewrite is True:
            with open(filepath, "wb") as fp:
                pickle.dump(self.xgbc_model, fp)