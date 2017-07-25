"""
Housing price tutorial for input functions.
Uses a regression approach.

@author Sam O | <samuel.ordonia@gmail.com>
"""
import itertools
import os

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Read inputs
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def predict_house_prices():
    # Create feature columns
    training_set, test_set, prediction_set = _read_inputs()
    feature_cols = [tf.contrib.layers.real_valued_column(feat) for feat in FEATURES]

    # Instantiate Regressor
    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[10, 10],
        model_dir='/tmp/housing_model'
    )

    # Build input fnc
    def input_fn(data_set):
        feature_cols = {feat: tf.constant(data_set[feat].values)
                            for feat in FEATURES}
        labels = tf.constant(data_set[LABEL].values)

        return feature_cols, labels

    # Train the regressor
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

    # Evaluate model
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev['loss']
    print('\nLoss:  {0:f}'.format(loss_score))

    # Predict housing median prices
    predictions = list(itertools.islice(
        regressor.predict(input_fn=lambda: input_fn(prediction_set)),
        6
    ))
    [print('Predicted median price:  {}'.format(p)) for p in predictions]
    print(prediction_set)


def _read_inputs():
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'references/housing_data'
    )

    training_set = pd.read_csv(os.path.join(data_path, 'boston_train.csv'), skipinitialspace=True, skiprows=1, names=COLUMNS)
    test_set = pd.read_csv(os.path.join(data_path, 'boston_test.csv'), skipinitialspace=True, skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv(os.path.join(data_path, 'boston_predict.csv'), skipinitialspace=True, skiprows=1, names=COLUMNS)

    return training_set, test_set, prediction_set
