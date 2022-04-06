import numpy as np
import pandas as pd
import sklearn.linear_model

import cmu_etim.base_regressor


class SmallRegressor(cmu_etim.base_regressor.BaseRegressor):
    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train a linear regression model

        Args:
            features (pd.DataFrame): a dataframe of features
            labels (pd.Series): a pandas column of labels

        """
        self.model = sklearn.linear_model.LinearRegression()
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the labels on new data

        Args:
            features (pd.DataFrame): a dataframe of features matching the training columns

        Returns:
            np.ndarray: the predictions for each recipe

        """
        return self.model.predict(features)


if __name__ == '__main__':
    regressor = SmallRegressor()
    tr, trl, tst, tstl = regressor.recipes()
    regressor.train(tr, trl['rating'])
    regressor.assess(tst, tstl['rating'], tstl['title'])
