"""
Basic tutorial on classification study on iris flowers.

Source:  <https://www.tensorflow.org/get_started/tflearn>

@author Sam O | <samuel.ordonia@gmail.com>
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
# import urllib
import requests
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf


# Data set URLs
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def iris_analysis():
    training_file, test_file = _download_iris_data(
        IRIS_TRAINING_URL,
        IRIS_TEST_URL
    )

    # Load data
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=training_file.name,
        target_dtype=np.int,
        features_dtype=np.float32
    )
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=test_file.name,
        target_dtype=np.int,
        features_dtype=np.float32
    )
    print('Data load successful')

    # Specify that all features have real-value data
    feature_columns = [
        tf.contrib.layers.real_valued_column(
            "",
            dimension=4
        )
    ]

    # Build 3-layer neural net
    hidden_units = [10, 20, 30]
    num_layers = 3
    if len(hidden_units) != num_layers:
        return 'Verify layer count & hidden unit vals'

    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        n_classes=num_layers,
        model_dir='/tmp/iris_model'
    )

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

    # Fit model
    num_steps = 2000
    classifier.fit(
        input_fn=get_train_inputs,
        steps=num_steps
    )
    print('Training on {} steps'.format(num_steps))

    # Eval accuracy
    accuracy_score = classifier.evaluate(
        input_fn=get_test_inputs,
        steps=1
    )['accuracy']
    print('\nTest accuracy:  {0: f}\n'.format(accuracy_score))

    # Close tmp files
    training_file.close()
    test_file.close()
    print('Closed files')


def _download_iris_data(IRIS_TRAINING_URL, IRIS_TEST_URL):
    """Grabs iris training and test csv's."""
    training_filename = os.path.basename(IRIS_TRAINING_URL)
    test_filename = os.path.basename(IRIS_TEST_URL)

    # Grab csv data & write to tempfiles
    training_file = NamedTemporaryFile(mode='w+')
    test_file = NamedTemporaryFile(mode='w+')

    # Write to tmp training file
    raw_training = requests.get(IRIS_TRAINING_URL)
    training_file.write(raw_training.text)
    training_file.seek(io.SEEK_SET)

    # Write to tmp test file
    raw_test = requests.get(IRIS_TRAINING_URL)
    test_file.write(raw_test.text)
    test_file.seek(io.SEEK_SET)

    return training_file, test_file


def _get_dataset_inputs(data_set):
    x = tf.constant(data_set.data)
    y = tf.constant(data_set.target)
    return x, y
