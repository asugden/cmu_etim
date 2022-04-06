import numpy as np
import pandas as pd
import sklearn.model_selection
from typing import Tuple


class BaseRegressor():
    def __init__(self):
        self.model = None
        self.columns = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Train a linear regression model

        Args:
            features (pd.DataFrame): a dataframe of features
            labels (pd.Series): a pandas column of labels

        """
        raise NotImplementedError

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the labels on new data

        Args:
            features (pd.DataFrame): a dataframe of features matching the training columns

        Returns:
            np.ndarray: the predictions for each recipe

        """
        raise NotImplementedError

    def assess(self, features: pd.DataFrame, labels: pd.Series, titles: pd.Series) -> float:
        """Assess the quality of the model

        Args:
            features (pd.DataFrame): a dataframe of features matching the training columns
            labels (pd.DataFrame): a dataframe of labels and recipe names, for fun

        Returns:
            float: the accuracy of the model

        """
        predicted_labels = self.predict(features)

        absolute_error = np.abs(labels - predicted_labels)
        mean_absolute_error = np.mean(absolute_error)
        print('Mean absolute error:', mean_absolute_error)
        print('Worst performing recipe',
              titles[np.argmax(absolute_error)])
        print('Should have been',
              labels[np.argmax(absolute_error)],
              'but turned out to be',
              predicted_labels[np.argmax(absolute_error)])

        return mean_absolute_error

    def recipes(self, seed: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get training recipes

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
        """
        df = pd.read_parquet('data/epicurious_recipes_clean.pq')

        labels = df[['rating', 'title']]
        features = df.drop(columns=['rating', 'title', '#cakeweek'])

        train_features, train_labels, test_features, test_labels = sklearn.model_selection.train_test_split(
            features, labels, test_size=0.3, random_state=seed)
        return (train_features.reset_index(drop=True),
                test_features.reset_index(drop=True),
                train_labels.reset_index(drop=True),
                test_labels.reset_index(drop=True))
