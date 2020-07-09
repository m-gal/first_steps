# -*- coding: utf-8 -*-
"""
Created on Feb 20 2020
    This code make the features preprocessing
    and learn the baseline feed-forward neuarl network
@author: mikhail.galkin
"""
# """
# # TO DO:
# [X] Move batch size from ds to model.fit.
# [X] ML flow tracking
#
# [X] Adjust the mlflow artifacts
# [X] Train on imbalanced data
#
# [X] Vanilla: grid search through:
#   [X] neurons#1
#   [X] bath size
#   [X] optimiser
# [X] Vanilla: Understand the BathNormalization place in net
#   [X] DenseFeatures -> BatchNormalization -> Dense -> BatchNormalization -> ...
#   [X] DenseFeatures -> Dense -> BatchNormalization -> ...
# [X] Get params for best vanilla for:
#   [X] bath normalization
#   [X] Tanh as hidden activation function
# """
#%% [markdown] -----------------------------------------------------------------
### Start running code
##### _ '----' in cell's title means printing snipet and may be non-mandatory for real_
##### _ '====' in cell's title means saving snipet and may be non-mandatory for real_
#%% Import TensorFlow and other libraries
from __future__ import absolute_import, division, print_function, unicode_literals

import time; start_time = time.time()
import tensorflow as tf             ;print('tensorflow ver.:', tf.__version__)
from tensorflow import keras        ;print('keras ver.:', keras.__version__)

tf.enable_eager_execution()

#%% Importing required packages
import os
import json
import pickle
import numpy as np              ;print('numpy ver.:', np.__version__)
import pandas as pd             ;print('pandas ver.:', pd.__version__)
import matplotlib.pyplot as plt #;print('matplotlib ver.:', matplotlib.__version__)
import mlflow                   ;print('mlflow ver.:', mlflow.__version__)

import inspect # for model inspection

import sm128_models as sm128m
import sm128_plots as sm128p

#%% Cell for re-importin changed sm128_models \ plots --------------------------
import importlib
importlib.reload(sm128m)
importlib.reload(sm128p)

# print(inspect.getsource(sm128m.fit_model))
# print(inspect.getsource(sm128m.ffnn_vanilla_batchNorm_ddbd))
# print(inspect.getsource(sm128m.ffnn_vanilla_batchNorm_dbdbd))
#%% Small cell for comparing data loaded ---------------------------------------
# import datacompy
# compare = datacompy.Compare(df_train, df_train_1, join_columns='id')
# print(compare.report())

#%% Get path and start timer
f = os.getcwd().replace(os.path.basename(os.getcwd()), '')
#%% Load column data dictionary
with open(f + 'data_in/tf/dic_cols.json', 'r') as fp:
    dic_cols = json.load(fp)
# View columns types -----------------------------------------------------------
dic_cols
#%% Load (Unpickling) tensorflow feature_columns
with open(f + 'data_in/tf/tf_feature_cols.txt', 'rb') as fp:
    feature_cols = pickle.load(fp)
# View features columns --------------------------------------------------------
print(len(feature_cols))
print(*feature_cols, sep='\n')
#%% Load validation data
start_time_load = time.time()
df_val = pd.read_csv(f + 'data_in/tf/dflow_val.csv', sep=',', decimal='.', header=0,
                parse_dates=[
                    'first_status_day_date'
                ],
                date_parser=lambda col: pd.to_datetime(col).strftime('%Y-%m-%d')
                , infer_datetime_format=True
                , low_memory=False
                , float_precision='round_trip'
)
# ------------------------------------------------------------------------------
print('Validation data loaded for --- %s seconds ---' % (time.time() - start_time_load))
#%% Load test data
start_time_load = time.time()
df_test = pd.read_csv(f + 'data_in/tf/dflow_test.csv', sep=',', decimal='.', header=0,
                parse_dates=[
                    'first_status_day_date'
                ],
                date_parser=lambda col: pd.to_datetime(col).strftime('%Y-%m-%d')
                , infer_datetime_format=True
                , low_memory=False
                , float_precision='round_trip'
)
# ------------------------------------------------------------------------------
print('Test data loaded for --- %s seconds ---' % (time.time() - start_time_load))
#%% Load train data
start_time_load = time.time()
df_train = pd.read_csv(f + 'data_in/tf/dflow_train.csv', sep=',', decimal='.', header=0,
                parse_dates=[
                    'first_status_day_date'
                ],
                date_parser=lambda col: pd.to_datetime(col).strftime('%Y-%m-%d')
                , infer_datetime_format=True
                , low_memory=False
                , float_precision='round_trip'
)
# ------------------------------------------------------------------------------
print('Train data loaded for --- %s seconds ---' % (time.time() - start_time_load))
#%% Load preprocessing parameters
start_time_load = time.time()
dflow_norm_params = pd.read_csv(f + 'data_in/tf/dflow_norm_params.csv', sep=',', decimal='.', header=0
                , index_col=0
                , low_memory=False
                , float_precision='round_trip'
)
#%% Load Embedding params as dictionary
with open(f + 'data_in/tf/dic_embe_params.json', 'r') as fp:
    dic_embe_params = json.load(fp)
# ------------------------------------------------------------------------------
print('All data loaded for --- %s seconds ---' % (time.time() - start_time_load))

#%% [markdown] -----------------------------------------------------------------
### Start to prepare data...
#### Normalizing the numeric data:
#%% For train data
df_train[dic_cols['x_nume']] -= dflow_norm_params.loc['min', :]
df_train[dic_cols['x_nume']] /= (dflow_norm_params.loc['max', :] - dflow_norm_params.loc['min', :])
#%% print ----------------------------------------------------------------------
print('Min-Max for TRAIN:', '\n', df_train['00_prev_total_profit'].iloc[97:111])
print('Min in TRAIN afrer Robust:', df_train['00_prev_total_profit'].min(), '\n',
        'Max in TRAIN afrer Robust:', df_train['00_prev_total_profit'].max())

#%% For validation data
df_val[dic_cols['x_nume']] -= dflow_norm_params.loc['min', :]
df_val[dic_cols['x_nume']] /= (dflow_norm_params.loc['max', :] - dflow_norm_params.loc['min', :])
#%% print ----------------------------------------------------------------------
print('Min-Max for VAL:', '\n', df_val['00_prev_total_profit'].iloc[97:111])
print('Min in VAL afrer Robust:', df_val['00_prev_total_profit'].min(), '\n',
        'Max in VAL afrer Robust:', df_val['00_prev_total_profit'].max())

#%% For test data
df_test[dic_cols['x_nume']] -= dflow_norm_params.loc['min', :]
df_test[dic_cols['x_nume']] /= (dflow_norm_params.loc['max', :] - dflow_norm_params.loc['min', :])
#%% print ----------------------------------------------------------------------
print('Min-Max for TEST:', '\n', df_test['00_prev_total_profit'].iloc[97:111])
print('Min in TEST afrer Robust:', df_test['00_prev_total_profit'].min(), '\n',
        'Max in TEST afrer Robust:', df_test['00_prev_total_profit'].max())



#%% [markdown] -----------------------------------------------------------------
#### DEMO block: Demonstrate several types of feature column
"""
#%% Understand the input pipeline ----------------------------------------------
# Now that we have created the input pipeline, let's call it to see the format
# of the data it returns. We have used a small batch size to keep the output readable.
# We can see that the dataset returns a dictionary of column names (from the dataframe)
# that map to column values from rows in the dataframe.
# A small batch sized is used for demonstration purposes
ds_val = df_to_ds(df_val, y='y0_1', batch_size=16)

for feature_batch, label_batch in ds_val.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['00_act_app_age'])
  print('A batch of city:', feature_batch['00_act_city'])
  print('A batch of loan_number:', feature_batch['00_act_loan_number'])
  print('A batch of targets:', label_batch )

#%% Function for demonstrate several types of feature column -------------------
# TensorFlow provides many types of feature columns.
# In this section, we will create several types of feature columns,
# and demonstrate how they transform a column from the dataframe.
## We will use this batch to demonstrate several types of feature columns
example_batch = next(iter(ds_val))[0]
# A utility method to create a feature column
# and to transform a batch of data
def demo(feature_column):
  feature_layer = keras.layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

#%% Numeric columns ------------------------------------------------------------
# The output of a feature column becomes the input to the model
# (using the demo function defined above, we will be able to see exactly
# how each column from the dataframe is transformed).
# A numeric column is the simplest type of column.
# It is used to represent real valued features.
# When using this column, your model will receive the column value from
# the dataframe unchanged.
age = feature_column.numeric_column('00_act_app_age')
demo(age)

#%% Bucketized columns ---------------------------------------------------------
# Often, you don't want to feed a number directly into the model,
# but instead split its value into different categories based on numerical ranges.
# Consider raw data that represents a person's age.
# Instead of representing age as a numeric column, we could split the age
# into several buckets using a bucketized column.
# Notice the one-hot values below describe which age range each row matches.
age_buckets = feature_column.bucketized_column(
    age,
    boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    )
demo(age_buckets)

#%% Categorical columns --------------------------------------------------------
# In this dataset, thal is represented as a string (e.g. city, email_domain).
# We cannot feed strings directly to a model.
# Instead, we must first map them to numeric values.
# The categorical vocabulary columns provide a way to represent strings
# as a one-hot vector (much like you have seen above with age buckets).
# The vocabulary can be passed as a list using categorical_column_with_vocabulary_list,
# or loaded from a file using categorical_column_with_vocabulary_file.
key = '00_act_ga_device_category'
print(key)
vocabulary_list = df_train[key].unique()
print(vocabulary_list)
device_category = feature_column.categorical_column_with_vocabulary_list(
    key=key,
    vocabulary_list=vocabulary_list
    )
device_category_one_hot = feature_column.indicator_column(device_category)
demo(device_category_one_hot)

#%% Embedding columns ----------------------------------------------------------
# Suppose instead of having just a few possible strings, we have thousands
# (or more) values per category. For a number of reasons, as the number
# of categories grow large, it becomes infeasible to train a neural network
# using one-hot encodings. We can use an embedding column to overcome this limitation.
# Instead of representing the data as a one-hot vector of many dimensions,
# an embedding column represents that data as a lower-dimensional,
# dense vector in which each cell can contain any number, not just 0 or 1.
# The size of the embedding (6, in the example below) is a parameter that must be tuned.
# Key point:
# using an embedding column is best when a categorical column has many possible values.
# Notice the input to the embedding column is the categorical column
# we previously created
device_category_embedding = feature_column.embedding_column(
    device_category,
    dimension=6
    )
demo(device_category_embedding)

#%% Hashed feature columns -----------------------------------------------------
# Another way to represent a categorical column with a large number of values
# is to use a categorical_column_with_hash_bucket.
# This feature column calculates a hash value of the input, then selects
# one of the hash_bucket_size buckets to encode a string.
# When using this column, you do not need to provide the vocabulary,
# and you can choose to make the number of hash_buckets significantly smaller
# than the number of actual categories to save space.
# Key point:
# An important downside of this technique is that there may be collisions
# in which different strings are mapped to the same bucket.
# In practice, this can work well for some datasets regardless.
device_category_hashed = feature_column.categorical_column_with_hash_bucket(
      key,
      hash_bucket_size=1000
      )
demo(feature_column.indicator_column(device_category_hashed))

#%% Crossed feature columns ----------------------------------------------------
# Combining features into a single feature, better known as feature crosses,
# enables a model to learn separate weights for each combination of features.
# Here, we will create a new feature that is the cross of age and device_category.
# Note that crossed_column does not build the full table of all possible combinations
# (which could be very large). Instead, it is backed by a hashed_column,
# so you can choose how large the table is.
crossed_feature = feature_column.crossed_column(
    [age_buckets, device_category],
    hash_bucket_size=1000
    )
demo(feature_column.indicator_column(crossed_feature))
"""
#%% [markdown]------------------------------------------------------------------
##### DEMO block fished

#%% Convert dataframe to dataset
start_time_convert = time.time()
ds_train = sm128m.df_to_ds(df=df_train, dic_cols=dic_cols, y='y0_1')
ds_val = sm128m.df_to_ds(df=df_val, dic_cols=dic_cols, y='y0_1')
ds_test = sm128m.df_to_ds(df=df_test, dic_cols=dic_cols, y='y0_1')
# ------------------------------------------------------------------------------
print("Data converted --- %s seconds ---" % (time.time() - start_time_convert),
    "\nAnd ready for training --- %s seconds ---" % (time.time() - start_time))
# %whos

#%% View some results of converted ds-------------------------------------------
for feature_batch, label_batch in ds_val.take(4):
  # print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['00_act_app_age'])
  print('A batch of city:', feature_batch['00_act_city'])
  print('A batch of loan_number:', feature_batch['00_act_loan_number'])
  print('A batch of targets:', label_batch )

#%% [markdown]------------------------------------------------------------------
##### Start Neural Network developing ------------------------------------------
#%% Calculate CLASS WEIGHTS
neg, pos = np.bincount(df_train["y0_1"])
total = neg + pos

# Scaling by (total/2) helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
# Variant#1 from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
weight_1_for_0 = (1 / neg) * (total) / 2.0
weight_1_for_1 = (1 / pos) * (total) / 2.0
# ! THIS APPROACH IS BETTER:
CLASS_WEIGHT_1 = {0: weight_1_for_0, 1: weight_1_for_1}

# Variant#2 from https://keras.io/examples/structured_data/imbalanced_classification/
weight_2_for_0 = 1 / neg
weight_2_for_1 = 1 / pos
# THIS IS NUMBER #2 AT APPROACHES:
CLASS_WEIGHT_2 = {0: weight_2_for_0, 1: weight_2_for_1}
# ^-----------------------------------------------------------------------------
print(
    "Train set:\n \
    Total: {}\n \
    Positive(1): {} ({:.2f}% of total)\n \
    Negative(0): {} ({:.2f}% of total)\n".format(
        total, pos, 100 * pos / total, neg, 100 * neg / total
    )
)

print(
    "Weight for class 1: var#1={:.2f}, var#2={:.6f}".format(
        weight_1_for_1, weight_2_for_1
    )
)
print(
    "Weight for class 0: var#1={:.2f}, var#2={:.6f}".format(
        weight_1_for_0, weight_2_for_0
    )
)

print(
    "\nValidation set:\n",
    "class 1: {} ({:.2f}%)\n".format(
        np.bincount(df_val["y0_1"])[1],
        np.bincount(df_val["y0_1"])[1] / len(df_val) * 100,
    ),
    "class 0: {} ({:.2f}%)".format(
        np.bincount(df_val["y0_1"])[0],
        np.bincount(df_val["y0_1"])[0] / len(df_val) * 100,
    ),
)
print("\n CLASS_WEIGHT_1 for exploiting:", CLASS_WEIGHT_1)
print(" CLASS_WEIGHT_2 for exploiting:", CLASS_WEIGHT_2)

#%% Get features parameters for FUNCTIONAL API
start_time_feat = time.time()
# feature_columns_api, feature_inputs_api, feature_layer_api = sm128m.get_feature_api(
#     dic_cols, dic_embe_params
# )
# for v in feature_inputs_api.values(): print(v, sep='\n')
feature_multi_columns_api, feature_multi_inputs_api = sm128m.get_feature_multi_api(
    dic_cols, dic_embe_params, n_loans=20
)
# ^-----------------------------------------------------------------------------
print(
    "\nFeatures inputs prepared --- %s seconds ---" % (time.time() - start_time_feat),
    "\nand ready to start training --- %s seconds ---" % (time.time() - start_time),
)


#%%[markdown]-------------------------------------------------------------------
#### Set MLFLOW tracking and TENSORBOARD callbacks
#%% Set TENSORBOARD CALLBACKS
LOG_DIR = f + "tensorboard\\fit\\"
# ^-----------------------------------------------------------------------------
print(
    f"You can find TensorBoard lods in {LOG_DIR}",
    "\n!!! DONT FOGET RUN NEW TERMINAL !!!",
    "\n> tensorboard --logdir=t://Documents/DataProjects/ml_ec/tensorboard/fit ",
)

#%% Set MLFLOW TRACKING
# The folder MUST BE named as 'mlruns' --runing only ONCE
TEMP_DIR = f + "temp\\"  # folder for temporary data
EXPERIMENT_PATH = "file:///T:/Documents/DataProjects/ml_ec/mlruns"
mlflow.set_tracking_uri(EXPERIMENT_PATH)
# ^-----------------------------------------------------------------------------
print(
    f"The experiment can be found at {EXPERIMENT_PATH}",  # " and has an ID of {EXPERIMENT_ID}",
    "\n!!! DONT FOGET RUN NEW TERMINAL !!!",
    "\n> mlflow ui ",
)

#%% DEFINE NAME for current experiment and create folder for it
EXPERIMENT_NAME = "cnn"
mlflow.set_experiment(EXPERIMENT_NAME)

#%% RUN EXPERIMENT *************************************************************
# PARAMETERS
SEED = 42
BATCH_SIZE = 128
NEURONS_1 = 256
NEURONS_2 = None  # 128
EPOCHS = 10
ACTIVATION_LAYERS = "relu"
ACTIVATION_RECURRENT = "sigmoid"
ACTIVATION_OUTPUT = "sigmoid"
OPTIMIZER = "adam"
LOSS = "binary_crossentropy"
METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]

CLASS_WEIGHT = None
DROPOUT_RATE = None
BATCH_NORM = None

NN_NAME = "parallel_SeparableConv_to_dense"

start_time_run = time.time()
for OPTIMIZER in ['adam', 'rmsprop', 'sgd']:
    for BATCH_SIZE in [64, 128, 512]:
        for NEURONS_1 in [64, 128, 512]:
            for NEURONS_2 in [64, 128]:
                for FILTERS_3 in [16, 32]:
                    for FILTERS_5 in [16, 32]:
                        for CLASS_WEIGHT [None, CLASS_WEIGHT_1, CLASS_WEIGHT_2]:

                            # if FILTERS_3 == 32 and FILTERS_5 == 32:
                            #     print(f"Skip over filters={FILTERS_3}x{FILTERS_5}")
                            #     continue

                            print("\n")
                            print(f"optimizer={OPTIMIZER} : batch_size={BATCH_SIZE}")
                            print(f"neurons={NEURONS_1}x{NEURONS_2}")
                            print(f"filters={FILTERS_3}x{FILTERS_5}")
                            print(f"class_weight={CLASS_WEIGHT}")

                            #! for reproducubility
                            np.random.seed(SEED)
                            python_random.seed(SEED)
                            tf.random.set_seed(SEED)

                            """
                                Use with mlflow.start_run: in the Python code to create a new MLflow run.
                                This is the recommended way to use MLflow in notebook cells.
                                Whether your code completes or exits with an error,
                                the with context will make sure to close the MLflow run,
                                so you don't have to call mlflow.end_run
                            """
                            with mlflow.start_run(
                                run_name=NN_NAME,
                            ) as run:

                                #! BUILD model
                                model = sm128m.parallel_conv_to_dense(
                                    feature_multi_columns_api,
                                    feature_multi_inputs_api,
                                    filters_3=FILTERS_3,
                                    filters_5=FILTERS_5,
                                    neurons_1=NEURONS_1,
                                    neurons_2=NEURONS_2,
                                    activation_layers=ACTIVATION_LAYERS,
                                    activation_output=ACTIVATION_OUTPUT,
                                    optimizer=OPTIMIZER,
                                    loss=LOSS,
                                    metrics=METRICS,
                                )

                                #! TRAIN model
                                current_time = time.strftime("%Y%m%d-%H%M")
                                filepath = (
                                    f
                                    + "models\\tf\\"
                                    + current_time
                                    + "-"
                                    + NN_NAME
                                    + "-best_mc.h5"
                                )

                                history = sm128m.fit_model(
                                    model=model,
                                    train_dataset=ds_train,
                                    validation_dataset=ds_val,
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS,
                                    verbose=1,
                                    # ! The class weights go here
                                    class_weight=CLASS_WEIGHT,
                                    callbacks=[
                                        tf.keras.callbacks.TensorBoard(
                                            log_dir=LOG_DIR + current_time
                                        ),
                                        tf.keras.callbacks.EarlyStopping(
                                            monitor="val_auc",
                                            patience=1,
                                            verbose=1,
                                            mode="max",
                                        ),
                                        tf.keras.callbacks.ModelCheckpoint(
                                            filepath=filepath,
                                            monitor="val_auc",
                                            verbose=1,
                                            mode="max",
                                            save_best_only=True,
                                        ),
                                    ],
                                )

                                #! EVALUATE model
                                train_eval = sm128m.eval_model(
                                    model,
                                    dataset=ds_train,
                                    batch_size=BATCH_SIZE,
                                    dataset_name="train",
                                )
                                val_eval = sm128m.eval_model(
                                    model,
                                    dataset=ds_val,
                                    batch_size=BATCH_SIZE,
                                    dataset_name="val",
                                )
                                test_eval = sm128m.eval_model(
                                    model,
                                    dataset=ds_test,
                                    batch_size=BATCH_SIZE,
                                    dataset_name="test",
                                )

                                #! MAKE prediction
                                train_predict = sm128m.predict_model(
                                    model,
                                    dataset=ds_train,
                                    batch_size=BATCH_SIZE,
                                    dataset_name="train",
                                )
                                val_predict = sm128m.predict_model(
                                    model,
                                    dataset=ds_val,
                                    batch_size=BATCH_SIZE,
                                    dataset_name="val",
                                )
                                test_predict = sm128m.predict_model(
                                    model,
                                    dataset=ds_test,
                                    batch_size=BATCH_SIZE,
                                    dataset_name="test",
                                )

                                #! SAVE prediction
                                np.save(TEMP_DIR + "train_predict.npy", train_predict)
                                np.save(TEMP_DIR + "val_predict.npy", val_predict)
                                np.save(TEMP_DIR + "test_predict.npy", test_predict)

                                #! PLOT results
                                # -------- history LOSS
                                plot_loss = sm128p.plot_loss(history, nn_name=NN_NAME)
                                plot_loss.savefig(TEMP_DIR + "plot_loss.png", dpi=100)
                                # -------- history METRICS
                                plot_metrics = sm128p.plot_metrics(history, nn_name=NN_NAME)
                                plot_metrics.savefig(TEMP_DIR + "plot_metrics.png", dpi=100)
                                # -------- CONFUSION MATRIX for validation set
                                plot_cm_val = sm128p.plot_cm(
                                    target=df_val["y0_1"],
                                    prediction=val_predict,
                                    data_name="Val set",
                                    nn_name=NN_NAME,
                                    p=0.5,
                                )
                                plot_cm_val.savefig(TEMP_DIR + "plot_cm_val.png", dpi=100)
                                # -------- CONFUSION MATRIX for test set
                                plot_cm_test = sm128p.plot_cm(
                                    target=df_test["y0_1"],
                                    prediction=test_predict,
                                    data_name="Test set",
                                    nn_name=NN_NAME,
                                    p=0.5,
                                )
                                plot_cm_test.savefig(TEMP_DIR + "plot_cm_test.png", dpi=100)
                                # -------- ROC curve
                                plot_3roc = sm128p.plot_3roc(
                                    train_target=df_train["y0_1"],
                                    train_prediction=train_predict,
                                    val_target=df_val["y0_1"],
                                    val_prediction=val_predict,
                                    test_target=df_test["y0_1"],
                                    test_prediction=test_predict,
                                    nn_name=NN_NAME,
                                )
                                plot_3roc.savefig(TEMP_DIR + "plot_3roc.png", dpi=100)
                                # -------- ROC curve for ALL \ NEW \ ECISTING
                                plot_9x3roc = sm128p.plot_9x3roc(
                                    dtrain=df_train[
                                        ["customer_type", "act_loan_numinstal", "y0_1"]
                                    ],
                                    train_prediction=train_predict,
                                    dval=df_val[
                                        ["customer_type", "act_loan_numinstal", "y0_1"]
                                    ],
                                    val_prediction=val_predict,
                                    dtest=df_test[
                                        ["customer_type", "act_loan_numinstal", "y0_1"]
                                    ],
                                    test_prediction=test_predict,
                                    nn_name=NN_NAME,
                                )
                                plot_9x3roc.savefig(TEMP_DIR + "plot_9x3roc.png", dpi=100)
                                # -------- score predicted DENSITY
                                plot_dens = sm128p.plot_dens(
                                    train_target=df_train["y0_1"],
                                    train_prediction=train_predict,
                                    test_target=df_test["y0_1"],
                                    test_prediction=test_predict,
                                    nn_name=NN_NAME,
                                )
                                plot_dens.savefig(TEMP_DIR + "plot_density.png", dpi=100)
                                # -------- model's CODE
                                print(
                                    inspect.getsource(sm128m.parallel_conv_to_dense),
                                    file=open(TEMP_DIR + "model.txt", "w"),
                                )
                                # ----------------------------------------------------------
                                #! LOG PARAMETERS
                                mlflow.log_param(
                                    "neurons", str(NEURONS_1) + "x" + str(NEURONS_2)
                                )
                                mlflow.log_param("optimizer", OPTIMIZER)
                                mlflow.log_param("loss", LOSS)
                                mlflow.log_param(
                                    "epochs",
                                    str(EPOCHS) + ":" + str(len(history.history["loss"])),
                                )
                                mlflow.log_param("batch_size", BATCH_SIZE)
                                mlflow.log_param("activ_layers", ACTIVATION_LAYERS)
                                mlflow.log_param("activ_output", ACTIVATION_OUTPUT)
                                mlflow.log_param(
                                    "conv1D",
                                    f"[{FILTERS_3}:3k,{FILTERS_5}:5k]:1strides:same",
                                )
                                mlflow.log_param("pool", "max:2size:2strides:valid")
                                mlflow.log_param("RNN", None)
                                mlflow.log_param("batchNorm", BATCH_NORM)
                                mlflow.log_param("dropout", DROPOUT_RATE)
                                mlflow.log_param("class_weight", CLASS_WEIGHT)
                                mlflow.log_param("seed", SEED)
                                mlflow.log_param("fnote", "pool[n]::merge:flat:dense")

                                #! log METRICS
                                for key, values in history.history.items():
                                    for v in values:
                                        print(key, v)
                                        mlflow.log_metric(key=key, value=v)
                                mlflow.log_metric(
                                    "diff_auc",
                                    history.history["auc"][-1]
                                    - history.history["val_auc"][-1],
                                )
                                mlflow.log_metric(
                                    "f1",
                                    2
                                    * (
                                        history.history["precision"][-1]
                                        * history.history["recall"][-1]
                                    )
                                    / (
                                        history.history["precision"][-1]
                                        + history.history["recall"][-1]
                                    ),
                                )

                                #! log PLOTS and model's CODE
                                mlflow.log_artifact(TEMP_DIR + "plot_loss.png")
                                mlflow.log_artifact(TEMP_DIR + "plot_metrics.png")
                                mlflow.log_artifact(TEMP_DIR + "plot_cm_test.png")
                                mlflow.log_artifact(TEMP_DIR + "plot_cm_val.png")
                                mlflow.log_artifact(TEMP_DIR + "plot_3roc.png")
                                mlflow.log_artifact(TEMP_DIR + "plot_9x3roc.png")
                                mlflow.log_artifact(TEMP_DIR + "plot_density.png")
                                mlflow.log_artifact(TEMP_DIR + "model.txt")

                                #! log DATA
                                mlflow.log_artifact(TEMP_DIR + "train_predict.npy")
                                mlflow.log_artifact(TEMP_DIR + "val_predict.npy")
                                mlflow.log_artifact(TEMP_DIR + "test_predict.npy")

                                #! log MODEL
                                model_file = (
                                    TEMP_DIR
                                    + time.strftime("%Y%m%d")
                                    + "-"
                                    + NN_NAME
                                    + "-"
                                    + "auc"
                                    + str(round(history.history["auc"][-1], 4))
                                    + "-"
                                    + "val_auc"
                                    + str(round(history.history["val_auc"][-1], 4))
                                    + ".h5"
                                )
                                model.save(model_file)
                                mlflow.log_artifact(model_file)

                                tf.keras.backend.clear_session()

print("====  VALIO !!! ====")
print(f"Run for --- {(time.time() - start_time_run)/60} minutes ---")
print(f"All for --- {(time.time() - start_time)/60} minutes ---")
winsound.Beep(frequency=2500, duration=1000)
# EXPERIMENT END ***************************************************************

#%% Print model summary
print(model.summary())
