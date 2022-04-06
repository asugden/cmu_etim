import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import cmu_etim.base_regressor


class LargeRegressor(cmu_etim.base_regressor.BaseRegressor):
    def create(self, n_features: int, layer_dims: tuple[int] = [500, 400, 300, 200, 100, 50, 20, 10, 5, 2]):
        """Create a new model

        Args:


        """
        feature_set = keras.Input(shape=(n_features, ))

        x = feature_set

        for layer_dim in layer_dims:
            x = keras.layers.Dense(
                layer_dim, activation='relu', use_bias=True)(x)
            x = keras.layers.Dropout(0.3)(x)

        x = keras.layers.Dense(1, activation='linear', use_bias=True)(x)

        # Initialize model
        self.model = keras.Model(inputs=feature_set, outputs=x)

    def train(self,
              features: pd.DataFrame,
              labels: pd.Series,
              loss_function: keras.losses.Loss = keras.losses.mean_squared_error,
              optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam,
              initial_lr: int = 0.001,
              epochs: int = 200):
        """Fit the model to data

        Args:
            train (np.ndarray): a training dataset
            labels (np.ndarray): the labels of the training dataset
            loss_function (keras.losses.Loss, optional): The loss function for training. Defaults to keras.losses.binary_crossentropy.
            optimizer (keras.optimizers.Optimizer, optional): The optimizer for training. Defaults to keras.optimizers.Adam.
            initial_lr (int, optional): Initial loss rate. Defaults to 0.0001.
            epochs (int, optional): number of training e. Defaults to 50.
            train_verbose (int, optional): [description]. Defaults to 1.
            rescale (bool, optional): If True, rescale data to be between 0 and 1
        """
        metrics = keras.metrics.MeanAbsoluteError()

        self.model.compile(optimizer=optimizer(
            learning_rate=initial_lr), loss=loss_function)
        self.model.fit(features.values.astype(float), labels.values, shuffle=True, epochs=epochs,
                       verbose=1, batch_size=64, validation_split=0.2)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the outputs given a model

        Args:
            test (np.ndarray): test data in the format of train data

        Returns:
            np.ndarray: The predictions per position
        """
        prob = self.model.predict(features.values.astype(float))
        pred = np.squeeze(prob, axis=1)

        return pred

    def save(self, path: str = 'data/saved_tf_model') -> None:
        """Save the model

        Args:
            path (str, optional): location to save model. Defaults to 'data/saved_tf_model'.
        """
        self.model.save(path)


if __name__ == '__main__':
    regressor = LargeRegressor()
    tr, trl, tst, tstl = regressor.recipes()
    # regressor.create(len(tr.columns))
    # regressor.train(tr, trl['rating'])
    # regressor.save()
    regressor.model = keras.models.load_model('data/saved_tf_model')
    regressor.assess(tst, tstl['rating'], tstl['title'])
