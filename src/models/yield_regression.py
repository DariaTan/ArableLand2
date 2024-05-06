import xgboost
from sklearn.metrics import mean_absolute_percentage_error


class regression():
    """Class including XGBRegression model and some methods relating to it
    """
    def __init__(self, n_estimators):
        self.xgbr_model = xgboost.XGBRegressor(n_estimators = n_estimators)
    def fit(self, X_train, Y_train):
        self.xgbr_model.fit(X_train, Y_train)
        return self.xgbr_model
    
    def r2(self, X, Y):
        R2 = self.xgbr_model.score(X, Y)
        return R2

    def predict(self, X):
        return self.xgbr_model.predict(X)

    def mape(self, Y_pred, Y_test):
        mape = mean_absolute_percentage_error(Y_pred, Y_test)
        return mape
    
    def fe(self):
        return self.xgbr_model.feature_importances_
    