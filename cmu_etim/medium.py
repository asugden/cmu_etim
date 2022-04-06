import numpy as np
import pandas as pd
import xgboost

import cmu_etim.base_regressor


class MediumRegressor(cmu_etim.base_regressor.BaseRegressor):
    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train a linear regression model

        Args:
            features (pd.DataFrame): a dataframe of features
            labels (pd.Series): a pandas column of labels

        """
        np.random.seed(7)
        self.columns = np.random.choice(features.columns, 50, replace=False)
        print(self.columns)
        features = features[self.columns]

        self.model = xgboost.XGBRegressor()
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the labels on new data

        Args:
            features (pd.DataFrame): a dataframe of features matching the training columns

        Returns:
            np.ndarray: the predictions for each recipe

        """
        features = features[self.columns]
        return self.model.predict(features)


if __name__ == '__main__':
    regressor = MediumRegressor()
    tr, trl, tst, tstl = regressor.recipes()
    regressor.train(tr, trl['rating'])
    regressor.assess(tst, tstl['rating'], tstl['title'])
