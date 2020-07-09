# -*- coding: utf-8 -*-
"""
Created at May 2020
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
# to read it from disk directly.
def df_to_ds(df, dic_cols, y="y0_1", batch_size=0, shuffle=False):
    labels = df[y]
    df = df[dic_cols["x"]].copy()
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


#%% Get TF#2.0  feature columns for the Keras Functional API
def get_feature_api(dic_cols, dic_embe_params):

    feature_columns_api = []
    feature_inputs_api = {}  # for KERAS Functional API

    for col in dic_cols["x"]:
        if col in dic_cols["x_nume"]:
            feature_columns_api.append(tf.feature_column.numeric_column(col))
            feature_inputs_api[col] = tf.keras.Input(shape=(1,), name=col)

        elif col in dic_cols["x_bina"]:
            feature_columns_api.append(tf.feature_column.numeric_column(col))
            feature_inputs_api[col] = tf.keras.Input(shape=(1,), name=col)

        elif col in dic_cols["x_indi"]:
            indi = tf.feature_column.categorical_column_with_vocabulary_list(
                col, ["m", "f"]
            )
            indi_ohe = tf.feature_column.indicator_column(indi)

            feature_columns_api.append(indi_ohe)
            feature_inputs_api[col] = tf.keras.Input(
                shape=(1,), name=col, dtype=tf.string
            )

        elif col in dic_cols["x_embe"]:
            vocabulary_list = dic_embe_params[col]["vocab"]
            dimension = dic_embe_params[col]["dim"]

            categorical = tf.feature_column.categorical_column_with_vocabulary_list(
                col, vocabulary_list=vocabulary_list
            )
            embedding = tf.feature_column.embedding_column(
                categorical, dimension=dimension
            )

            feature_columns_api.append(embedding)
            feature_inputs_api[col] = tf.keras.Input(
                shape=(1,), name=col, dtype=tf.string
            )

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns_api)
    feature_layer_api = feature_layer(feature_inputs_api)

    return feature_columns_api, feature_inputs_api, feature_layer_api


#%% Function: FIT model
def fit_model(model, train_dataset, validation_dataset, batch_size, **kwargs):
    # fit model
    history = model.fit(
        train_dataset.batch(batch_size),
        validation_data=validation_dataset.batch(batch_size),
        **kwargs,
    )
    return history


#%% Function: EVALUATE model
def eval_model(model, dataset, batch_size, dataset_name, verbose=1):
    # evaluate model
    print("\nModel evaluation for: " + dataset_name)
    evaluation = model.evaluate(dataset.batch(batch_size), verbose=verbose)
    print("\nEvaluation results for: " + dataset_name)
    for name, value in zip(model.metrics_names, evaluation):
        print(name, ": ", value)
    return evaluation


#%% Function: PREDICT model
def predict_model(model, dataset, batch_size, dataset_name):
    print("\nModel prediction for: " + dataset_name)
    prediction = model.predict(dataset.batch(batch_size))
    print(dataset_name + " predicted")
    return prediction


#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~ M O D E L S ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#%% VANILLA FEED-FORWARD
def ffnn_vanilla(
    feature_cols,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
):

    model = keras.models.Sequential(
        [
            keras.layers.DenseFeatures(feature_cols),
            keras.layers.Dense(units=neurons, activation=activation_layers),
            keras.layers.Dense(1, activation=activation_output),
        ]
    )
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model


#%% VANILLA FEED-FORWARD WITH FUNCTIONAL API
def ffnn_vanilla_api(
    feature_inputs_api,
    feature_layer_api,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
):

    # Build model with Keras Functional API
    x = tf.keras.layers.Dense(
        units=neurons, activation=activation_layers, name="dense_1"
    )(feature_layer_api)
    outputs = tf.keras.layers.Dense(1, activation=activation_output, name="y0_1")(x)

    model = tf.keras.Model(
        inputs=[v for v in feature_inputs_api.values()],
        outputs=outputs,
        name="vanilla_api",
    )
    model.compile(optimizer, loss, metrics)
    model.reset_states()
    return model


#%% VANILLA FEED-FORWARD WITH FUNCTIONAL API AND MULTI (LOANS) INPUTS
def ffnn_multi_vanilla_api(
    feature_multi_columns_api,
    feature_multi_inputs_api,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    print_summary=False,
):
    feature_multi_layers = {}
    # dense1 = {}
    for n_loan in reversed(range(0, 20)):
        n = str(n_loan).zfill(2)
        # print(n)
        feature_multi_layer = tf.keras.layers.DenseFeatures(
            feature_multi_columns_api[n], name="dense_features_" + n
        )
        feature_multi_layers[n] = feature_multi_layer(feature_multi_inputs_api[n])

    merge = tf.keras.layers.Concatenate(name="merge")(
        list(feature_multi_layers.values())
    )
    x = tf.keras.layers.Dense(
        units=neurons, activation=activation_layers, name="dense1"
    )(merge)
    outputs = tf.keras.layers.Dense(1, activation=activation_output, name="y0_1")(x)

    model = tf.keras.Model(
        inputs=[v for v in feature_multi_inputs_api.values()],
        outputs=outputs,
        name="multi_vanilla_api",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model


#%%  MULTI DENSE WITH FUNCTIONAL API AND MULTI (LOANS) INPUTS
def ffnn_multi_dense_api(
    feature_multi_columns_api,
    feature_multi_inputs_api,
    neurons,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    dropout_rate=0.0,
    print_summary=False,
    **kwarg,
):
    feature_multi_layers = {}
    # dense1 = {}
    for n_loan in reversed(range(0, 20)):
        n = str(n_loan).zfill(2)
        # print(n)
        feature_multi_layer = tf.keras.layers.DenseFeatures(
            feature_multi_columns_api[n], name="dense_features_" + n
        )
        feature_multi_layers[n] = feature_multi_layer(feature_multi_inputs_api[n])
        # dense1[n] = n

    merge = tf.keras.layers.Concatenate(name="merge")(
        list(feature_multi_layers.values())
    )

    x = tf.keras.layers.Dense(
        units=neurons, activation=activation_layers, name="dense1"
    )(merge)
    x = tf.keras.layers.Dropout(rate=dropout_rate, seed=42)(x)
    x = tf.keras.layers.Dense(
        units=neurons, activation=activation_layers, name="dense2"
    )(x)
    outputs = tf.keras.layers.Dense(1, activation=activation_output, name="y0_1")(x)

    model = tf.keras.Model(
        inputs=[v for v in feature_multi_inputs_api.values()],
        outputs=outputs,
        name="multi_vanilla_api",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model


#%%  MULTI CONVOLUTIONAL WITH FUNCTIONAL API AND MULTI (LOANS) INPUTS
def multi_cnn_inputs(
    feature_multi_columns_api,
    feature_multi_inputs_api,
    neurons_1,
    neurons_2,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    print_summary=False,
    **kwarg,
):
    feature_multi_layers = {}
    reshape_layers = {}
    conv_layers = {}
    batch_layers = {}
    conv_activation_layers = {}
    pool_layers = {}
    flatt_layers = {}

    for n_loan in reversed(range(0, 20)):
        if n_loan == 0:
            n_shape = 279
        else:
            n_shape = 281

        n = str(n_loan).zfill(2)
        # print(n)
        feature_multi_layer = tf.keras.layers.DenseFeatures(
            feature_multi_columns_api[n], name="dense_features_" + n
        )
        feature_multi_layers[n] = feature_multi_layer(feature_multi_inputs_api[n])
        reshape_layers[n] = tf.keras.layers.Reshape((n_shape, 1), name="reshape_" + n)(
            feature_multi_layers[n]
        )
        conv_layers[n] = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            # activation="relu",
            input_shape=(None, n_shape, 1),
            name="conv_" + n,
        )(reshape_layers[n])
        batch_layers[n] = tf.keras.layers.BatchNormalization()(conv_layers[n])
        conv_activation_layers[n] = tf.keras.layers.Activation(activation="relu")(
            batch_layers[n]
        )
        pool_layers[n] = tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2, padding="valid", name="pool_" + n
        )(conv_activation_layers[n])
        flatt_layers[n] = tf.keras.layers.Flatten(name="flatten_" + n)(pool_layers[n])

    merge = tf.keras.layers.Concatenate(name="merge")(list(flatt_layers.values()))

    x = tf.keras.layers.Dense(
        units=neurons_1, activation=activation_layers, name="dense1"
    )(merge)
    x = tf.keras.layers.Dense(
        units=neurons_2, activation=activation_layers, name="dense2"
    )(x)
    outputs = tf.keras.layers.Dense(1, activation=activation_output, name="y0_1")(x)

    model = tf.keras.Model(
        inputs=[v for v in feature_multi_inputs_api.values()],
        outputs=outputs,
        name="multi_cnn_inputs",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model


#%%  Inception blocks for all loans and with CNN
def parallel_conv_to_cnn(
    feature_multi_columns_api,
    feature_multi_inputs_api,
    neurons_1,
    neurons_2,
    activation_layers,
    activation_output,
    optimizer,
    loss,
    metrics,
    print_summary=False,
    **kwarg,
):
    feature_multi_layers = {}
    reshape_layers = {}
    conv3_layers = {}
    conv5_layers = {}
    merge_conv_layers = {}
    pool_layers = {}

    loans = [str(n).zfill(2) for n in reversed(range(0, 20))]

    for n in loans:
        if n == "00":
            n_shape = 279
        else:
            n_shape = 281

        feature_multi_layer = tf.keras.layers.DenseFeatures(
            feature_multi_columns_api[n], name="dense_features_" + n
        )
        feature_multi_layers[n] = feature_multi_layer(feature_multi_inputs_api[n])

        reshape_layers[n] = tf.keras.layers.Reshape((n_shape, 1), name="reshape_" + n)(
            feature_multi_layers[n]
        )
        # Padding for active loan
        if n == "00":
            reshape_layers[n] = tf.keras.layers.ZeroPadding1D(
                padding=1, name="zero_padded_" + n
            )(reshape_layers[n])

        #! 1D convolution with 3 kernel_size
        conv3_layers[n] = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(None, n_shape, 1),
            name="conv3_" + n,
        )(reshape_layers[n])
        #! 1D convolution with 5 kernel_size
        conv5_layers[n] = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(None, n_shape, 1),
            name="conv5_" + n,
        )(reshape_layers[n])

        #! stack all gotten filters to one tensor and max polling them
        merge_conv_layers[n] = tf.keras.layers.Concatenate(name="merge_conv_" + n)(
            [conv3_layers[n], conv5_layers[n]]
        )
        pool_layers[n] = tf.keras.layers.MaxPooling1D(
            pool_size=2, strides=2, padding="valid", name="pool_" + n
        )(merge_conv_layers[n])

    #! Concat pooled filters for all loans and before feeding them into RNN
    merge = tf.keras.layers.Concatenate(name="merge")(list(pool_layers.values()))
    #! Reccurent layer
    rnn = tf.keras.layers.LSTM(units=neurons_1, activation="tanh", name="rnn")(merge)
    x = tf.keras.layers.Dense(
        units=neurons_2, activation=activation_layers, name="dense1"
    )(rnn)
    outputs = tf.keras.layers.Dense(1, activation=activation_output, name="y0_1")(x)

    model = tf.keras.Model(
        inputs=[v for v in feature_multi_inputs_api.values()],
        outputs=outputs,
        name="parallel_conv_to_cnn",
    )
    model.compile(optimizer, loss, metrics)

    if print_summary:
        print(model.summary())

    model.reset_states()
    return model
