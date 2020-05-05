# -*- coding: utf-8 -*-
"""
Created at Mar 2020
    This code contains the models (as a rule Keras)
    and some functions for model development
@author: mikhail.galkin
"""
#%% Importing required packages
import tensorflow as tf
from tensorflow import keras

################################################################################
############################ F U N C T I O N S #################################
################################################################################

#%% Create an input pipeline using tf.data
# Next, we will wrap the dataframes with tf.data.
# This will enable us to use feature columns as a bridge to map from the columns
# in the Pandas dataframe to features used to train the model.
# If we were working with a very large CSV file
# (so large that it does not fit into memory), we would use tf.data
# to read it from disk directly. That is not covered in this tutorial.
def df_to_ds(df, dic_cols, y='y0_1', batch_size=0, shuffle=False):
    labels = df[y]
    df = df[dic_cols['x']].copy()
    # The given tensors are sliced along their first dimension.
    # This operation preserves the structure of the input tensors,
    # removing the first dimension of each tensor and using it as the dataset dimension.
    # All input tensors must have the same size in their first dimensions.
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    # Combines consecutive elements of this dataset into batches.
    if batch_size != 0:
        ds = ds.batch(batch_size)
    return ds

#%% Function: FIT model
def fit_model(model, train_dataset, validation_dataset, batch_size, **kwargs):
    # fit model
    history = model.fit(
        train_dataset.batch(batch_size),
        validation_data=validation_dataset.batch(batch_size),
        **kwargs)
    return history

#%% Function: EVALUATE model
def eval_model(model, dataset, batch_size, dataset_name, verbose=1):
    # evaluate model
    print('\nModel evaluation for: ' + dataset_name)
    evaluation = model.evaluate(dataset.batch(batch_size), verbose=verbose)
    print('\nEvaluation results for: ' + dataset_name)
    for name, value in zip(model.metrics_names, evaluation):
        print(name, ': ', value)
    return evaluation

#%% Function: PREDICT model
def predict_model(model, dataset, batch_size, dataset_name):
    print('\nModel prediction for: ' + dataset_name)
    prediction = model.predict(dataset.batch(batch_size))
    print(dataset_name + ' predicted')
    return prediction

################################################################################
############################# M O D E L S ######################################
################################################################################
#%% VANILLA FEED-FORWARD
def ffnn_vanilla(feature_cols, neurons,
    activation_layers, activation_output,
    optimizer, loss, metrics):

    model = keras.models.Sequential([
        keras.layers.DenseFeatures(feature_cols)
        , keras.layers.Dense(units=neurons, activation=activation_layers)
        , keras.layers.Dense(1, activation=activation_output)
        ])
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model

#%% VANILLA 3 DENSE
def ffnn_3Dense(feature_cols, neurons, neurons_2, neurons_3,
    activation_layers, activation_output,
    optimizer, loss, metrics):

    model = keras.models.Sequential([
        keras.layers.DenseFeatures(feature_cols)
        , keras.layers.Dense(units=neurons, activation=activation_layers)
        , keras.layers.Dense(units=neurons_2, activation=activation_layers)
        , keras.layers.Dense(units=neurons_3, activation=activation_layers)
        , keras.layers.Dense(1, activation=activation_output)
        ])
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model

#%% VANILLA 2 DENSE + Batch Normalization
def ffnn_2Dense_batchNorm(feature_cols, neurons, neurons_2,
    activation_layers, activation_output,
    optimizer, loss, metrics):

    model = keras.models.Sequential([
        keras.layers.DenseFeatures(feature_cols)
        , keras.layers.BatchNormalization()
        , keras.layers.Dense(units=neurons, activation=activation_layers)
        , keras.layers.BatchNormalization()
        , keras.layers.Dense(units=neurons_2, activation=activation_layers)
        , keras.layers.BatchNormalization()
        , keras.layers.Dense(1, activation=activation_output)
        ])
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model

#%% VANILLA 2 DENSE + Batch Normalization + Dropout
def ffnn_2Dense2Drop_batchNorm(feature_cols, neurons, neurons_2,
    activation_layers, activation_output,
    optimizer, loss, metrics):

    model = keras.models.Sequential([
        keras.layers.DenseFeatures(feature_cols)
        , keras.layers.Dense(units=neurons, activation=activation_layers)
        , keras.layers.Dropout(rate=0.2)
        , keras.layers.BatchNormalization()
        , keras.layers.Dense(units=neurons_2, activation=activation_layers)
        , keras.layers.Dropout(rate=0.2)
        , keras.layers.BatchNormalization()
        , keras.layers.Dense(1, activation=activation_output)
        ])
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model

#%% LSTM VANILLA
def lstm_vanilla(feature_cols, neurons,
    activation_layers, activation_output, activation_recurrent,
    optimizer, loss, metrics,
    **kwargs):

    model = keras.models.Sequential([
        keras.layers.DenseFeatures(feature_cols)
        , keras.layers.LSTM(units=neurons, activation=activation_layers,
                            recurrent_activation=activation_recurrent)
        , keras.layers.BatchNormalization()
        , keras.layers.Dense(1, activation=activation_output)
        ])
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model
