import numpy as np
import pandas as pd
import sklearn.metrics
from typing import Tuple
import xgboost


class MediumCake():
    def __init__(self):
        self.model = None
        self.test_features = None
        self.test_titles = None

    def set_test_data(self, test_features: pd.Series, test_titles: pd.Series) -> None:
        """Set the testing data for lookups

        Args:
            test_features (pd.Series): _description_
            test_titles (pd.Series): _description_
        """
        self.test_features = test_features
        self.test_titles = test_titles.str.lower().str.strip()

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train a binary XGBoost classifier

        Args:
            features (pd.DataFrame): a dataframe of features
            labels (pd.Series): a pandas column of labels

        """
        self.model = xgboost.XGBClassifier(scale_pos_weight=1000)
        self.model.fit(features, labels)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the labels on new data

        Args:
            features (pd.DataFrame): a dataframe of features matching the training columns

        Returns:
            np.ndarray: the predictions for each recipe

        """
        return self.model.predict(features.astype(float))

    def predict_title(self, title: str) -> bool:
        """Predict the type of a recipe by name

        Args:
            title (str): the name of a recipe

        Returns:
            bool: True if it is a cake, False otherwise

        """
        features = self.test_features.loc[(
            self.test_titles == title).idxmax(), :].values.reshape((1, -1))
        return self.predict(features)[0] > 0

    def recipes(self, seed: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get training recipes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame,
                pd.DataFrame, pd.DataFrame]: _description_
        """
        df = pd.read_parquet('data/epicurious_recipes_clean.pq')

        labels = df[['title', '#cakeweek']]
        features = df.drop(columns=['title', '#cakeweek'])

        train_features, train_labels, test_features, test_labels = sklearn.model_selection.train_test_split(
            features, labels, test_size=0.5, random_state=seed)
        return (train_features.reset_index(drop=True),
                test_features.reset_index(drop=True),
                train_labels.reset_index(drop=True),
                test_labels.reset_index(drop=True))


if __name__ == '__main__':
    regressor = MediumCake()
    tr, trl, tst, tstl = regressor.recipes()
    regressor.set_test_data(tst, tstl['title'])

    regressor.train(pd.concat([tr, tst]), pd.concat(
        [trl['#cakeweek'], tstl['#cakeweek']]))

    for i, title in enumerate(tstl['title']):
        features = regressor.test_features.iloc[i, :].values.reshape((1, -1))
        if regressor.predict(features)[0] > 0:
            print(title)
