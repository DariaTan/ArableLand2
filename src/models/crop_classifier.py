from xgboost import XGBClassifier
import pickle
import sys, os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# from catboost import CatboostClassifier
from sklearn.model_selection import GridSearchCV

sys.path.append(os.path.dirname(os.getcwd()))
from src import utils


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

    def fit_classifier(self, model_path, type):
        """Fits the model if .pkl file doesn't exist

        Returns:
        --------
            self.xgbc_model.fit
        """
        rewrite=False
        if self.loaded_model is None:
            if type=="XGB":
                rewrite=True
                self.xgbc_model.fit(self.X_train, self.y_train, verbose=3)
            elif type=="all":
                names = [
                        # "Logistic Regression",
                        # "Random Forest",
                        # "Naive Bayes",
                        # "MLP Classifier",
                        "AdaBoost",
                        # "CatBoost"
                    ]
                classifiers = [
                        # LogisticRegression(),
                        # RandomForestClassifier(),
                        # GaussianNB(),
                        # MLPClassifier(),
                        AdaBoostClassifier(estimator=DecisionTreeClassifier()),
                        # CatboostClassifier()
                    ]
                params = [
                    # {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]},
                    # {'n_estimators':[20, 100], 'max_depth':[2,3]},
                    # {'var_smoothing': np.logspace(0,-9, num=10)},
                    # {'solver': ['adam'], 'max_iter': [1000], 'alpha': [0.1, 0.001]},
                    {'n_estimators':[20], 'algorithm': ["SAMME"], 'learning_rate':[0.01,0.1]}#'base_estimator__max_depth':[2,4], 'n_estimators':[20], },
                ]
                for name, param, clf in zip(names, params, classifiers):
                    print(name)
                    gs = GridSearchCV(
                        estimator=clf,
                        param_grid=param,
                        scoring='f1')
                    self.xgbc_model = gs.fit(self.X_train, self.y_train)
                    self.scores()
            else:
                print("Set type to xgb or all")
            if rewrite:
                with open(model_path, "wb") as fp:
                    pickle.dump(self.xgbc_model, fp)
                print("Model saved")
                self.scores()

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
        
        # with open(f'{}.txt', 'w') as file:
        #     file.write(pickle.dumps(self.scores))
        for k, v in self.scores.items():
            print(k, '=', v)
            
        # print("\nBaseline")
        # self.scores_baseline = utils.xgbc_scores_binary(self.xgbc_model, 
        #                                        self.X_test, self.y_test,
        #                                        baseline=True)
        # for k, v in self.scores_baseline.items():
        #     print(k, '=', v)
        return self.scores
    
    
    def save(self, filepath, rewrite=False):
        """
        Saves prefitted model to .pkl file

        Parameters:
        --------
            filepath: Callable[str]
                path to save file at
            rewrite: Optional[Bool]
                flag, if True, rewrites file
        """
        
