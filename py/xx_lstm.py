# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 2019
@author: mikhail.galkin

This code do LSTM reccurent neural network for forecasting portfolio quality

TO DO:
    [x] Use Normalizing instead MinMaxScaling :
    [x] Try not scale the {(t-1)Target}, but only macroindicies
    [x] Use ONLY forecasted variables from data provider
    [x] Try use TimeSeries cross-validation
"""
# --------------------------------------------------------------------------------------------------
# Import required libraries and modules
# --------------------------------------------------------------------------------------------------
import tensorflow as tf

tf.set_random_seed(42)  # For reproducibility

from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LeakyReLU  # advanced activation
from keras.layers import PReLU  # advanced activation

# Load needed libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Set up needed options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_info_columns", 500)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, \
#                            write_grads=False, write_images=False, embeddings_freq=0,\
#                            embeddings_layer_names=None, embeddings_metadata=None, \
#                            embeddings_data=None, update_freq='epoch')

# import specified functions and models
from LSTM_functions import ts_to_supervised
from LSTM_functions import calcPredict
from LSTM_functions import viewModelMAPE
from LSTM_functions import viewPredict

from LSTM_models import lstm_fit
from LSTM_models import dense_vanilla
from LSTM_models import dense_stacked
from LSTM_models import lstm_vanilla
from LSTM_models import lstm_stacked
from LSTM_models import lstm_bidirect
from LSTM_models import lstm_bidirect_stacked
from LSTM_models import cnn_dense
from LSTM_models import cnn_2dense

# --------------------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------------------
f = "C:/Users/mikhail.galkin/Documents/DataProjects/py_LSTM/data_in/data_ts.csv"
df = pd.read_csv(f, sep=",", decimal=".", header=0)

# Rename columns
df.info(verbose=True)
old_names = list(df.columns)
old_names
new_names = [n.replace("pol_", "") for n in old_names]
new_names
df.columns = new_names
df.info(verbose=True)

# Change type for date
df.date_ym.dtypes
df.date_ym = df.date_ym.astype(str)
df.date_ym.dtypes
df.date_ym.describe(include="all")

del (f, new_names, old_names)
# --------------------------------------------------------------------------------------------------
# Cleaning data
# --------------------------------------------------------------------------------------------------
# Delete not usful columns
df = df.drop(["target_1_30", "target_1_60", "target_2_30"], axis=1)
df.info(verbose=True)

# Remove zero-variance features
from sklearn.feature_selection import VarianceThreshold

df_num = df.select_dtypes(include=["float64", "int64"])
train_vars = df_num.loc[:, df_num.columns != "target"]
zero_var_filter = VarianceThreshold(threshold=0)
zero_var_filter.fit(train_vars)
zero_vars = list(
    set(train_vars.columns) - set(train_vars.columns[zero_var_filter.get_support()])
)
set(train_vars.columns[zero_var_filter.get_support()])
zero_vars
# Delete Zero-variance variables
df = df.drop(zero_vars, axis=1)
df.info()
df.describe().T

del (VarianceThreshold, df_num, train_vars, zero_var_filter, zero_vars)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
01: CREATE several different datasets for supervised problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
test_size = 0.15
shuffle = True  # for random data split
# shuffle=False # for last point split=FALSE
random_state = 951

# --------------------------------------------------------------------------------------------------
# (t-1): Shift dataset with times to supervised onto 1 times lag.
df_shift_1 = ts_to_supervised(df, 1, 1)
df_shift.info()
df_shift_1[["(t)date_ym", "(t)target", "(t-1)date_ym", "(t-1)target"]].head(10)

# drop columns we don't want to predict
names_shift_1 = list("(t-1)" + df.columns)
names_shift_1.remove("(t-1)date_ym")
names_shift_1.extend(["(t)target", "(t)date_ym"])

df_shift_1 = df_shift_1[names_shift_1]
df_shift_1.info()
df_shift_1[
    [
        "(t)date_ym",
        "(t)target",
        "(t)consumer_credit_m",
        "(t-1)target",
        "(t-1)consumer_credit_m",
    ]
].head(10)
df_shift_1[["(t)date_ym", "(t)target", "(t-1)target"]].tail(10)
df_shift_1[["(t)target", "(t-1)target"]].plot(figsize=(14, 7))
del names_shift_1
# --------------------------------------------------------------------------------------------------
# (t-1): RANDOMLY split Train & Test datasets
# split on train-test
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    df_shift_1.drop(["(t)date_ym", "(t)target"], axis=1),
    df_shift_1["(t)target"],
    test_size=test_size,
    shuffle=shuffle,
    random_state=random_state,
)
# view test data
df_shift_1.loc[y_test_1.index, ["(t)date_ym", "(t)target"]].sort_values(
    by=["(t)date_ym"]
)
# prepare full dataset
X_1 = df_shift_1.drop(["(t)date_ym", "(t)target"], axis=1)
y_1 = df_shift_1["(t)target"]
# --------------------------------------------------------------------------------------------------
# (t-1): Use Min-Max Scaling and Reshaping data
# ** Normalizing data ** Works more worse than minMax scaling
# scale the data
X_train_scaled_1 = scaler.fit_transform(X_train_1)
X_test_scaled_1 = scaler.transform(X_test_1)
X_scaled_1 = scaler.transform(X_1)

# reshape input to be 3D [samples, timesteps, features]
X_train_scaled_3d_1 = X_train_scaled_1.reshape(
    (X_train_scaled_1.shape[0], 1, X_train_scaled_1.shape[1])
)
X_test_scaled_3d_1 = X_test_scaled_1.reshape(
    (X_test_scaled_1.shape[0], 1, X_test_scaled_1.shape[1])
)
X_scaled_3d_1 = X_scaled_1.reshape((X_scaled_1.shape[0], 1, X_scaled_1.shape[1]))

print(
    X_train_scaled_3d_1.shape,
    y_train_1.shape,
    X_test_scaled_3d_1.shape,
    y_test_1.shape,
    X_scaled_3d_1.shape,
    y_1.shape,
)

# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): Shift dataset with 2 times to supervised
# --------------------------------------------------------------------------------------------------
df_shift_2 = ts_to_supervised(df, 2, 1)
# df_shift_2.info()
df_shift_2[
    [
        "(t)date_ym",
        "(t)target",
        "(t-1)date_ym",
        "(t-1)target",
        "(t-2)date_ym",
        "(t-2)target",
    ]
].head(10)

# drop columns we don't want to predict
names_shift_2 = list("(t-2)" + df.columns)
names_shift_2.extend(list("(t-1)" + df.columns))
names_shift_2 = [e for e in names_shift_2 if e not in ("(t-2)date_ym", "(t-1)date_ym")]
names_shift_2.extend(["(t)target", "(t)date_ym"])

df_shift_2 = df_shift_2[names_shift_2]
df_shift_2.info()
df_shift_2[["(t)date_ym", "(t)target", "(t-1)target", "(t-2)target"]].head(10)
df_shift_2[["(t)date_ym", "(t)target", "(t-1)target", "(t-2)target"]].tail(10)
df_shift_2[["(t)target", "(t-1)target", "(t-2)target"]].plot(figsize=(14, 7))
del names_shift_2
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1):Split Train & Test datasets
# split on train-test
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    df_shift_2.drop(["(t)date_ym", "(t)target"], axis=1),
    df_shift_2["(t)target"],
    test_size=test_size,
    shuffle=shuffle,
    random_state=random_state,
)
# view test data
df_shift_2.loc[y_test_2.index, ["(t)date_ym", "(t)target"]].sort_values(
    by=["(t)date_ym"]
)
# prepare full dataset
X_2 = df_shift_2.drop(["(t)date_ym", "(t)target"], axis=1)
y_2 = df_shift_2["(t)target"]
# X_2.info()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1):Min-Max Scaling and Reshaping data
# scale the data
X_train_scaled_2 = scaler.fit_transform(X_train_2)
X_test_scaled_2 = scaler.transform(X_test_2)
X_scaled_2 = scaler.transform(X_2)

# reshape input to be 3D [samples, timesteps, features]
X_train_scaled_3d_2 = X_train_scaled_2.reshape(
    (X_train_scaled_2.shape[0], 1, X_train_scaled_2.shape[1])
)
X_test_scaled_3d_2 = X_test_scaled_2.reshape(
    (X_test_scaled_2.shape[0], 1, X_test_scaled_2.shape[1])
)
X_scaled_3d_2 = X_scaled_2.reshape((X_scaled_2.shape[0], 1, X_scaled_2.shape[1]))

print(
    X_train_scaled_3d_2.shape,
    y_train_2.shape,
    X_test_scaled_3d_2.shape,
    y_test_2.shape,
    X_scaled_3d_2.shape,
    y_2.shape,
)

# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1): Shift dataset with 3 times to supervised: Shown best reulst with basic config
# --------------------------------------------------------------------------------------------------
df_shift_3 = ts_to_supervised(df, 3, 1)
# df_shift_3.info()
df_shift_3[
    [
        "(t)date_ym",
        "(t)target",
        "(t-1)date_ym",
        "(t-1)target",
        "(t-2)date_ym",
        "(t-2)target",
        "(t-3)date_ym",
        "(t-3)target",
    ]
].head(10)

# drop columns we don't want to predict
names_shift_3 = list("(t-3)" + df.columns)
names_shift_3.extend(list("(t-2)" + df.columns))
names_shift_3.extend(list("(t-1)" + df.columns))
names_shift_3 = [
    e
    for e in names_shift_3
    if e not in ("(t-3)date_ym", "(t-2)date_ym", "(t-1)date_ym")
]
names_shift_3.extend(["(t)target", "(t)date_ym"])

df_shift_3 = df_shift_3[names_shift_3]
df_shift_3.info()
# df_shift_3[['(t)date_ym', '(t)target', '(t-1)target', '(t-2)target', '(t-3)target']].head(10)
df_shift_3[["(t)target", "(t-1)target", "(t-2)target", "(t-3)target"]].plot(
    figsize=(14, 7)
)
del names_shift_3
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1):Split Train & Test datasets
# split on train-test
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    df_shift_3.drop(["(t)date_ym", "(t)target"], axis=1),
    df_shift_3["(t)target"],
    test_size=test_size,
    shuffle=shuffle,
    random_state=random_state,
)
# view test data
df_shift_3.loc[y_test_3.index, ["(t)date_ym", "(t)target"]].sort_values(
    by=["(t)date_ym"]
)
# prepare full dataset
X_3 = df_shift_3.drop(["(t)date_ym", "(t)target"], axis=1)
y_3 = df_shift_3["(t)target"]
# X_3.info()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1):Min-Max Scaling and Reshaping data
# scale the data
X_train_scaled_3 = scaler.fit_transform(X_train_3)
X_test_scaled_3 = scaler.transform(X_test_3)
X_scaled_3 = scaler.transform(X_3)

# reshape input to be 3D [samples, timesteps, features]
X_train_scaled_3d_3 = X_train_scaled_3.reshape(
    (X_train_scaled_3.shape[0], 1, X_train_scaled_3.shape[1])
)
X_test_scaled_3d_3 = X_test_scaled_3.reshape(
    (X_test_scaled_3.shape[0], 1, X_test_scaled_3.shape[1])
)
X_scaled_3d_3 = X_scaled_3.reshape((X_scaled_3.shape[0], 1, X_scaled_3.shape[1]))

print(
    X_train_scaled_3d_3.shape,
    y_train_3.shape,
    X_test_scaled_3d_3.shape,
    y_test_3.shape,
    X_scaled_3d_3.shape,
    y_3.shape,
)

# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1): Shift dataset with 4 times to supervised
# --------------------------------------------------------------------------------------------------
df_shift_4 = ts_to_supervised(df, 4, 1)
# df_shift_4.info()
df_shift_4[
    [
        "(t)date_ym",
        "(t)target",
        "(t-1)date_ym",
        "(t-1)target",
        "(t-2)date_ym",
        "(t-2)target",
        "(t-3)date_ym",
        "(t-3)target",
        "(t-4)date_ym",
        "(t-4)target",
    ]
].head(10)

# drop columns we don't want to predict
names_shift_4 = list("(t-4)" + df.columns)
names_shift_4.extend(list("(t-3)" + df.columns))
names_shift_4.extend(list("(t-2)" + df.columns))
names_shift_4.extend(list("(t-1)" + df.columns))
names_shift_4 = [
    e
    for e in names_shift_4
    if e not in ("(t-4)date_ym", "(t-3)date_ym", "(t-2)date_ym", "(t-1)date_ym")
]
names_shift_4.extend(["(t)target", "(t)date_ym"])

df_shift_4 = df_shift_4[names_shift_4]
df_shift_4.info()
df_shift_4[
    ["(t)target", "(t-1)target", "(t-2)target", "(t-3)target", "(t-4)target"]
].plot(figsize=(14, 7))
del names_shift_4
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1):Split Train & Test datasets
# split on train-test
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(
    df_shift_4.drop(["(t)date_ym", "(t)target"], axis=1),
    df_shift_4["(t)target"],
    test_size=test_size,
    shuffle=shuffle,
    random_state=random_state,
)
df_shift_4[
    [
        "(t)date_ym",
        "(t)target",
        "(t-1)target",
        "(t-1)consumer_credit_m",
        "(t-2)target",
        "(t-1)consumer_credit_m",
        "(t-3)target",
        "(t-3)consumer_credit_m",
    ]
].head(10)
# view test data
df_shift_4.loc[y_test_4.index, ["(t)date_ym", "(t)target"]].sort_values(
    by=["(t)date_ym"]
)
# prepare full dataset
X_4 = df_shift_4.drop(["(t)date_ym", "(t)target"], axis=1)
y_4 = df_shift_4["(t)target"]
X_4.info()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1):Min-Max Scaling and Reshaping data
# scale the data
X_train_scaled_4 = scaler.fit_transform(X_train_4)
X_test_scaled_4 = scaler.transform(X_test_4)
X_scaled_4 = scaler.transform(X_4)

# reshape input to be 3D [samples, timesteps, features]
X_train_scaled_3d_4 = X_train_scaled_4.reshape(
    (X_train_scaled_4.shape[0], 1, X_train_scaled_4.shape[1])
)
X_test_scaled_3d_4 = X_test_scaled_4.reshape(
    (X_test_scaled_4.shape[0], 1, X_test_scaled_4.shape[1])
)
X_scaled_3d_4 = X_scaled_4.reshape((X_scaled_4.shape[0], 1, X_scaled_4.shape[1]))

print(
    X_train_scaled_3d_4.shape,
    y_train_4.shape,
    X_test_scaled_3d_4.shape,
    y_test_4.shape,
    X_scaled_3d_4.shape,
    y_4.shape,
)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
02: BUILD DENSE VANILLA NN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_dense = pd.DataFrame()
result_range = 100
# --------------------------------------------------------------------------------------------------
# (t-1): VANILLA DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_1
y_train = y_train_1
testX = X_test_scaled_1
y_test = y_test_1
devX = X_scaled_1
y = y_1

# fit VANILLA DENSE
model = dense_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_dense_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)VANILLA DENSE",
]
## view process
viewModelMAPE(model_dense_1, figsize=(14, 4))
viewPredict(model_dense_1, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "DENSE 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense.loc[r, "(t-1)train_mape"] = train_mape
    result_dense.loc[r, "(t-1)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): VANILLA DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_2
y_train = y_train_2
testX = X_test_scaled_2
y_test = y_test_2
devX = X_scaled_2
y = y_2

# fit VANILLA DENSE
model = dense_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_dense_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)VANILLA DENSE",
]
## view process
viewModelMAPE(model_dense_2, figsize=(14, 4))
viewPredict(model_dense_2, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "DENSE 2: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense.loc[r, "(t-2)..train_mape"] = train_mape
    result_dense.loc[r, "(t-2)..test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1): VANILLA DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_3
y_train = y_train_3
testX = X_test_scaled_3
y_test = y_test_3
devX = X_scaled_3
y = y_3

# fit VANILLA DENSE
model = dense_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_dense_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)VANILLA DENSE",
]
## view process
viewModelMAPE(model_dense_3, figsize=(14, 4))
viewPredict(model_dense_3, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "DENSE 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense.loc[r, "(t-3)..train_mape"] = train_mape
    result_dense.loc[r, "(t-3)..test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1): VANILLA DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_4
y_train = y_train_4
testX = X_test_scaled_4
y_test = y_test_4
devX = X_scaled_4
y = y_4

# fit VANILLA DENSE
model = dense_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_dense_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4)VANILLA DENSE",
]
## view process
viewModelMAPE(model_dense_4, figsize=(14, 4))
viewPredict(model_dense_4, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "DENSE 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense.loc[r, "(t-4)..train_mape"] = train_mape
    result_dense.loc[r, "(t-4)..test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
print("DONE")
# view results
viewModelMAPE(model_dense_1)
viewPredict(model_dense_1)
viewModelMAPE(model_dense_2)
viewPredict(model_dense_2)
viewModelMAPE(model_dense_3)
viewPredict(model_dense_3)
viewModelMAPE(model_dense_4)
viewPredict(model_dense_4)
# View the result
print(result_dense.describe().T, "Results for VANILLA DENSE")
result_dense.to_csv("result_dense.csv")
result_dense[result_dense != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# fig.savefig('result_vanilla.png')
# --------------------------------------------------------------------------------------------------
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
03: BUILD STACKED DENSE NN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_dense_st = pd.DataFrame()
result_range = 100
# --------------------------------------------------------------------------------------------------
# (t-1): STACKED DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_1
y_train = y_train_1
testX = X_test_scaled_1
y_test = y_test_1
devX = X_scaled_1
y = y_1

# fit VANILLA DENSE
model = dense_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_densest_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)STACKED 3 DENSE",
]
## view process
viewModelMAPE(model_densest_1, figsize=(14, 4))
viewPredict(model_densest_1, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED DENSE 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense_st.loc[r, "(t-1)train_mape"] = train_mape
    result_dense_st.loc[r, "(t-1)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): STACKED DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_2
y_train = y_train_2
testX = X_test_scaled_2
y_test = y_test_2
devX = X_scaled_2
y = y_2

# fit VANILLA DENSE
model = dense_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_densest_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)STACKED 3 DENSE",
]
## view process
viewModelMAPE(model_densest_2, figsize=(14, 4))
viewPredict(model_densest_2, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED DENSE 2: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense_st.loc[r, "(t-2)train_mape"] = train_mape
    result_dense_st.loc[r, "(t-2)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1): STACKED DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_3
y_train = y_train_3
testX = X_test_scaled_3
y_test = y_test_3
devX = X_scaled_3
y = y_3

# fit VANILLA DENSE
model = dense_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_densest_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)STACKED 3 DENSE",
]
## view process
viewModelMAPE(model_densest_3, figsize=(14, 4))
viewPredict(model_densest_3, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "(in loop)STACKED DENSE 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense_st.loc[r, "(in loop)(t-3)train_mape"] = train_mape
    result_dense_st.loc[r, "(in loop)(t-3)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1): STACKED DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_4
y_train = y_train_4
testX = X_test_scaled_4
y_test = y_test_4
devX = X_scaled_4
y = y_4

# fit VANILLA DENSE
model = dense_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_densest_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4)STACKED 3 DENSE",
]
## view process
viewModelMAPE(model_densest_4, figsize=(14, 4))
viewPredict(model_densest_4, figsize=(14, 4))

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED DENSE 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_dense_st.loc[r, "(t-4)train_mape"] = train_mape
    result_dense_st.loc[r, "(t-4)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
print("DONE DENSE STACKED")
# view results
viewModelMAPE(model_densest_1)
viewPredict(model_densest_1)
viewModelMAPE(model_densest_2)
viewPredict(model_densest_2)
viewModelMAPE(model_densest_3)
viewPredict(model_densest_3)
viewModelMAPE(model_densest_4)
viewPredict(model_densest_4)
# View the result
print(result_dense_st.describe().T, "Results for STACKED DENSE")
result_dense_st.to_csv("result_dense_st.csv")
result_dense_st[result_dense_st != 100].dropna().boxplot(figsize=(14, 7), rot=90)

print(
    result_dense_st.describe().T.filter(like="loop", axis=0),
    "Results for STACKED DENSE",
)
# fig.savefig('result_vanilla.png')
# --------------------------------------------------------------------------------------------------

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
04: BUILD VANILLA LSTM, beanchmark models.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_vanilla = pd.DataFrame()
result_range = 100
# --------------------------------------------------------------------------------------------------
# (t-1): VANILLA LSTM model
# --------------------------------------------------------------------------------------------------
# Assign the train, test dataset
trainX = X_train_scaled_3d_1
y_train = y_train_1
testX = X_test_scaled_3d_1
y_test = y_test_1
devX = X_scaled_3d_1
y = y_1

# fit VANILLA
model = lstm_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_vanilla_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)VANILLA LSTM",
]
## view process
viewModelMAPE(model_vanilla_1)
viewPredict(model_vanilla_1)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "VANILLA 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_vanilla.loc[r, "(t-1)train_mape"] = train_mape
    result_vanilla.loc[r, "(t-1)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): VANILLA LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-2)(t-1) shifted set
trainX = X_train_scaled_3d_2
y_train = y_train_2
testX = X_test_scaled_3d_2
y_test = y_test_2
devX = X_scaled_3d_2
y = y_2

# fit VANILLA
model = lstm_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_vanilla_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)VANILLA LSTM",
]
## view process
viewModelMAPE(model_vanilla_2)
viewPredict(model_vanilla_2)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "VANILLA 2: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_vanilla.loc[r, "(t-2)..train_mape"] = train_mape
    result_vanilla.loc[r, "(t-2)..test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1): VANILLA LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_3
y_train = y_train_3
testX = X_test_scaled_3d_3
y_test = y_test_3
devX = X_scaled_3d_3
y = y_3

# fit VANILLA
model = lstm_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_vanilla_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)VANILLA LSTM",
]
## view process
viewModelMAPE(model_vanilla_3)
viewPredict(model_vanilla_3)

# collect performance for current trainset configuratio
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "VANILLA 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_vanilla.loc[r, "(t-3)..train_mape"] = train_mape
    result_vanilla.loc[r, "(t-3)..test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1): VANILLA LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-4)(t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_4
y_train = y_train_4
testX = X_test_scaled_3d_4
y_test = y_test_4
devX = X_scaled_3d_4
y = y_4

# fit VANILLA
model = lstm_vanilla(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_vanilla_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4)VANILLA LSTM",
]
## view process
viewModelMAPE(model_vanilla_4)
viewPredict(model_vanilla_4)

# collect performance for current trainset configuratio
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "VANILLA 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_vanilla.loc[r, "(t-4)..train_mape"] = train_mape
    result_vanilla.loc[r, "(t-4)..test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
print("VANILLA LSTM DONE!!")
# view results
viewModelMAPE(model_vanilla_1)
viewPredict(model_vanilla_1)
viewModelMAPE(model_vanilla_2)
viewPredict(model_vanilla_2)
viewModelMAPE(model_vanilla_3)
viewPredict(model_vanilla_3)
viewModelMAPE(model_vanilla_4)
viewPredict(model_vanilla_4)
# View the result
print(result_vanilla.describe().T, "Results for VANILLA")
result_vanilla.to_csv("result_vanilla.csv")
result_vanilla[result_vanilla != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# fig.savefig('result_vanilla.png')
# --------------------------------------------------------------------------------------------------

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
05: BUILD STACKED LSTM.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_stacked = pd.DataFrame()
result_range = 100
# --------------------------------------------------------------------------------------------------
# (t-1): STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
trainX = X_train_scaled_3d_1
y_train = y_train_1
testX = X_test_scaled_3d_1
y_test = y_test_1
devX = X_scaled_3d_1
y = y_1

# fit STACKED
model = lstm_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_stack_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)STACKED LSTM",
]
## view process
viewModelMAPE(model_stack_1)
viewPredict(model_stack_1)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_stacked.loc[r, "(t-1)train_mape"] = train_mape
    result_stacked.loc[r, "(t-1)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-2)(t-1) shifted set
trainX = X_train_scaled_3d_2
y_train = y_train_2
testX = X_test_scaled_3d_2
y_test = y_test_2
devX = X_scaled_3d_2
y = y_2

# fit STACKED
model = lstm_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_stack_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)STACKED LSTM",
]
## view process
viewModelMAPE(model_stack_2)
viewPredict(model_stack_2)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED 2:Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_stacked.loc[r, "(t-2)-train_mape"] = train_mape
    result_stacked.loc[r, "(t-2)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1):STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_3
y_train = y_train_3
testX = X_test_scaled_3d_3
y_test = y_test_3
devX = X_scaled_3d_3
y = y_3

# fit STACKED
model = lstm_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_stack_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)STACKED LSTM",
]
## view process
viewModelMAPE(model_stack_3)
viewPredict(model_stack_3)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_stacked.loc[r, "(t-3)-train_mape"] = train_mape
    result_stacked.loc[r, "(t-3)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1):STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-4)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_4
y_train = y_train_4
testX = X_test_scaled_3d_4
y_test = y_test_4
devX = X_scaled_3d_4
y = y_4

# fit STACKED
model = lstm_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_stack_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4)STACKED LSTM",
]
## view process
viewModelMAPE(model_stack_4)
viewPredict(model_stack_4)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "STACKED 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_stacked.loc[r, "(t-4)-train_mape"] = train_mape
    result_stacked.loc[r, "(t-4)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
print("STACKED LSTM DONE!")
# view results
viewModelMAPE(model_stack_1)
viewPredict(model_stack_1)
viewModelMAPE(model_stack_2)
viewPredict(model_stack_2)
viewModelMAPE(model_stack_3)
viewPredict(model_stack_3)
viewModelMAPE(model_stack_4)
viewPredict(model_stack_4)
# View the result
print(result_stacked.describe().T, "Results for STACKED")
result_stacked.to_csv("result_stacked.csv")
result_stacked[result_stacked != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# fig.savefig('result_stacked.png')
# --------------------------------------------------------------------------------------------------
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
06: BUILD BIDIRECTIONAL LSTM.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_bidirect = pd.DataFrame()
result_range = 100
# --------------------------------------------------------------------------------------------------
# (t-1): BIDIRECTIONAL LSTM model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
trainX = X_train_scaled_3d_1
y_train = y_train_1
testX = X_test_scaled_3d_1
y_test = y_test_1
devX = X_scaled_3d_1
y = y_1

# fit BIDIRECTIONAL
model = lstm_bidirect(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bidirect_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)BIDIRECTIONAL LSTM",
]
## view process
viewModelMAPE(model_bidirect_1)
viewPredict(model_bidirect_1)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bidirect.loc[r, "(t-1)train_mape"] = train_mape
    result_bidirect.loc[r, "(t-1)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): BIDIRECTIONAL LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-2)(t-1) shifted set
trainX = X_train_scaled_3d_2
y_train = y_train_2
testX = X_test_scaled_3d_2
y_test = y_test_2
devX = X_scaled_3d_2
y = y_2

# fit BIDIRECTIONAL
model = lstm_bidirect(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bidirect_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)BIDIRECTIONAL LSTM",
]
## view process
viewModelMAPE(model_bidirect_2)
viewPredict(model_bidirect_2)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL 2: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bidirect.loc[r, "(t-2)-train_mape"] = train_mape
    result_bidirect.loc[r, "(t-2)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1):BIDIRECTIONAL LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_3
y_train = y_train_3
testX = X_test_scaled_3d_3
y_test = y_test_3
devX = X_scaled_3d_3
y = y_3

# fit BIDIRECTIONAL
model = lstm_bidirect(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bidirect_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)BIDIRECTIONAL LSTM",
]
## view process
viewModelMAPE(model_bidirect_3)
viewPredict(model_bidirect_3)

# collect performance for current trainset configuratio
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bidirect.loc[r, "(t-3)-train_mape"] = train_mape
    result_bidirect.loc[r, "(t-3)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1):BIDIRECTIONAL LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-4)(t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_4
y_train = y_train_4
testX = X_test_scaled_3d_4
y_test = y_test_4
devX = X_scaled_3d_4
y = y_4

# fit BIDIRECTIONAL
model = lstm_bidirect(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bidirect_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4)BIDIRECTIONAL LSTM",
]
## view process
viewModelMAPE(model_bidirect_4)
viewPredict(model_bidirect_4)

# collect performance for current trainset configuratio
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bidirect.loc[r, "(t-4)-train_mape"] = train_mape
    result_bidirect.loc[r, "(t-4)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
print("BIDIRECTIONAL LSTM DONE!")
# view results
viewModelMAPE(model_bidirect_1)
viewPredict(model_bidirect_1)
viewModelMAPE(model_bidirect_2)
viewPredict(model_bidirect_2)
viewModelMAPE(model_bidirect_3)
viewPredict(model_bidirect_3)
viewModelMAPE(model_bidirect_4)
viewPredict(model_bidirect_4)
# View the result
print(result_bidirect.describe().T, "Results for BIDIRECTIONAL")
result_bidirect.to_csv("result_bidirect.csv")
result_bidirect[result_bidirect != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# fig.savefig('result_bidirect.png')
# --------------------------------------------------------------------------------------------------
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
07: BUILD BIDIRECTIONAL STACKED LSTM.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_bistacked = pd.DataFrame()
result_range = 100
# --------------------------------------------------------------------------------------------------
# (t-1): BIDIRECTIONAL STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
trainX = X_train_scaled_3d_1
y_train = y_train_1
testX = X_test_scaled_3d_1
y_test = y_test_1
devX = X_scaled_3d_1
y = y_1

# fit BIDIRECTIONAL STACKED
model = lstm_bidirect_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bistacked_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)BIDIRECTIONAL STACKED LSTM",
]
# view process
viewModelMAPE(model_bistacked_1)
viewPredict(model_bistacked_1)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL STACKED 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bistacked.loc[r, "(t-1)train_mape"] = train_mape
    result_bistacked.loc[r, "(t-1)test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): BIDIRECTIONAL STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-2)(t-1) shifted set
trainX = X_train_scaled_3d_2
y_train = y_train_2
testX = X_test_scaled_3d_2
y_test = y_test_2
devX = X_scaled_3d_2
y = y_2

# fit BIDIRECTIONAL STACKED
model = lstm_bidirect_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bistacked_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)BIDIRECTIONAL STACKED LSTM",
]
# view process
viewModelMAPE(model_bistacked_2)
viewPredict(model_bistacked_2)

# collect performance for current trainset configuration
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL STACKED 2: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bistacked.loc[r, "(t-2)-train_mape"] = train_mape
    result_bistacked.loc[r, "(t-2)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1):BIDIRECTIONAL STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_3
y_train = y_train_3
testX = X_test_scaled_3d_3
y_test = y_test_3
devX = X_scaled_3d_3
y = y_3

# fit BIDIRECTIONAL STACKED
model = lstm_bidirect_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bistacked_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)BIDIRECTIONAL STACKED LSTM",
]
# view process
viewModelMAPE(model_bistacked_3)
viewPredict(model_bistacked_3)

# collect performance for current trainset configuratio
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL STACKED 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bistacked.loc[r, "(t-3)-train_mape"] = train_mape
    result_bistacked.loc[r, "(t-3)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1):BIDIRECTIONAL STACKED LSTM model
# --------------------------------------------------------------------------------------------------
# For (t-4)(t-3)(t-2)(t-1) shifted set
trainX = X_train_scaled_3d_4
y_train = y_train_4
testX = X_test_scaled_3d_4
y_test = y_test_4
devX = X_scaled_3d_4
y = y_4

# fit BIDIRECTIONAL STACKED
model = lstm_bidirect_stacked(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_bistacked_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4) BIDIRECTIONAL STACKED LSTM",
]
# view process
viewModelMAPE(model_bistacked_4)
viewPredict(model_bistacked_4)

# collect performance for current trainset configuratio
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "BIDIRECTIONAL STACKED 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_bistacked.loc[r, "(t-4)-train_mape"] = train_mape
    result_bistacked.loc[r, "(t-4)-test_mape"] = test_mape

K.clear_session()
# --------------------------------------------------------------------------------------------------
print("BIDIRECTIONAL STACKED LSTM DONE!")
# view results
viewModelMAPE(model_bistacked_1)
viewPredict(model_bistacked_1)
viewModelMAPE(model_bistacked_2)
viewPredict(model_bistacked_2)
viewModelMAPE(model_bistacked_3)
viewPredict(model_bistacked_3)
viewModelMAPE(model_bistacked_4)
viewPredict(model_bistacked_4)
# View the result
print(result_bistacked.describe().T, "Results for BIDIRECTIONAL STACKED")
result_bistacked.to_csv("result_bistacked.csv")
result_bistacked[result_bistacked != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# fig.savefig('result_bistacked.png')
# --------------------------------------------------------------------------------------------------
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XX: BUILD CONVOLUTIONAL NETS.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# create DF for collecting results about train set shapes
result_cnn_dense = pd.DataFrame()
result_range = 100
cnn_type = "2ConvPool2ConvDense"
# --------------------------------------------------------------------------------------------------
# (t-1): CONVOLUTIONAL DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
# reshape from [samples, timesteps] into [samples, timesteps, features]
trainX = X_train_scaled_1.reshape(
    X_train_scaled_1.shape[0], X_train_scaled_1.shape[1], 1
)
y_train = y_train_1
testX = X_test_scaled_1.reshape(X_test_scaled_1.shape[0], X_test_scaled_1.shape[1], 1)
y_test = y_test_1
devX = X_scaled_1.reshape(X_scaled_1.shape[0], X_scaled_1.shape[1], 1)
y = y_1
print(trainX.shape)

# fit CONVOLUTIONAL
model = cnn_dense(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_cnn_1 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-1)" + cnn_type,
]
# view process
viewModelMAPE(model_cnn_1, figsize=(14, 4))
viewPredict(model_cnn_1, figsize=(14, 4))

# collect performance
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        cnn_type
        + " 1: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_cnn_dense.loc[r, cnn_type + "(t-1)train_mape"] = train_mape
    result_cnn_dense.loc[r, cnn_type + "(t-1)test_mape"] = test_mape
K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-2)(t-1): CONVOLUTIONAL DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
# reshape from [samples, timesteps] into [samples, timesteps, features]
trainX = X_train_scaled_2.reshape(
    X_train_scaled_2.shape[0], X_train_scaled_2.shape[1], 1
)
y_train = y_train_2
testX = X_test_scaled_2.reshape(X_test_scaled_2.shape[0], X_test_scaled_2.shape[1], 1)
y_test = y_test_2
devX = X_scaled_2.reshape(X_scaled_2.shape[0], X_scaled_2.shape[1], 1)
y = y_2
print(trainX.shape)

# fit CONVOLUTIONAL
model = cnn_dense(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_cnn_2 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-2)" + cnn_type,
]
# view process
viewModelMAPE(model_cnn_2, figsize=(14, 4))
viewPredict(model_cnn_2, figsize=(14, 4))

# collect performance
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        cnn_type
        + " 2: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_cnn_dense.loc[r, cnn_type + "(t-2)train_mape"] = train_mape
    result_cnn_dense.loc[r, cnn_type + "(t-2)test_mape"] = test_mape
K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1): CONVOLUTIONAL DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
# reshape from [samples, timesteps] into [samples, timesteps, features]
trainX = X_train_scaled_3.reshape(
    X_train_scaled_3.shape[0], X_train_scaled_3.shape[1], 1
)
y_train = y_train_3
testX = X_test_scaled_3.reshape(X_test_scaled_3.shape[0], X_test_scaled_3.shape[1], 1)
y_test = y_test_3
devX = X_scaled_3.reshape(X_scaled_3.shape[0], X_scaled_3.shape[1], 1)
y = y_3
print(trainX.shape)

# fit CONVOLUTIONAL
model = cnn_dense(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_cnn_3 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-3)" + cnn_type,
]
# view process
viewModelMAPE(model_cnn_3, figsize=(14, 4))
viewPredict(model_cnn_3, figsize=(14, 4))

# collect performance
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        cnn_type
        + " 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_cnn_dense.loc[r, cnn_type + "(t-3)train_mape"] = train_mape
    result_cnn_dense.loc[r, cnn_type + "(t-3)test_mape"] = test_mape
K.clear_session()
# --------------------------------------------------------------------------------------------------
# (t-4)(t-3)(t-2)(t-1): CONVOLUTIONAL DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
# reshape from [samples, timesteps] into [samples, timesteps, features]
trainX = X_train_scaled_4.reshape(
    X_train_scaled_4.shape[0], X_train_scaled_4.shape[1], 1
)
y_train = y_train_4
testX = X_test_scaled_4.reshape(X_test_scaled_4.shape[0], X_test_scaled_4.shape[1], 1)
y_test = y_test_4
devX = X_scaled_4.reshape(X_scaled_4.shape[0], X_scaled_4.shape[1], 1)
y = y_4
print(trainX.shape)

# fit CONVOLUTIONAL
model = cnn_dense(trainX)
history = lstm_fit(trainX, y_train, model)
# make prediction
pred, score_train, score_test, score_dev = calcPredict(
    df, model, trainX, y_train, testX, y_test, devX, y
)
model_cnn_4 = [
    model,
    history,
    pred,
    score_train,
    score_test,
    score_dev,
    "(t-4)" + cnn_type,
]
# view process
viewModelMAPE(model_cnn_4, figsize=(14, 4))
viewPredict(model_cnn_4, figsize=(14, 4))

# collect performance
for r in range(result_range):
    lstm_fit(trainX, y_train, model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        cnn_type
        + " 4: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_cnn_dense.loc[r, cnn_type + "(t-4)train_mape"] = train_mape
    result_cnn_dense.loc[r, cnn_type + "(t-4)test_mape"] = test_mape
K.clear_session()
# --------------------------------------------------------------------------------------------------
print(cnn_type + " DONE!")
# view results
viewModelMAPE(model_cnn_1)
viewPredict(model_cnn_1)
viewModelMAPE(model_cnn_2)
viewPredict(model_cnn_2)
viewModelMAPE(model_cnn_3)
viewPredict(model_cnn_3)
viewModelMAPE(model_cnn_4)
viewPredict(model_cnn_4)
# View the result
print(
    result_cnn_dense.describe().T.filter(like=cnn_type, axis=0),
    "Results for " + cnn_type,
)
print(
    result_cnn_dense.describe().T.filter(like="test", axis=0), "Results for " + cnn_type
)
result_cnn_dense.to_csv("result_cnn_dense.csv")
result_cnn_dense[result_cnn_dense != 100].dropna().boxplot(figsize=(14, 7), rot=90)

# fig.savefig('result_vanilla.png')
# -------------------------------------------------------------------------------------------------


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FIND OPTIMAL PARAMETERS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
## STACKED DENSE (t-3) ligth tuning
# assign
trainX = X_train_scaled_3
y_train = y_train_3
testX = X_test_scaled_3
y_test = y_test_3
devX = X_scaled_3
y = y_3

# DENSE STACKED: Experiment for numbers of EPOCHS and NEURONS and DROPOUT
result_end = pd.DataFrame()
grid_epochs = [50]
grid_neurons = [300, 500, 1000]  # 300
grid_dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
repeats = 100

for e in grid_epochs:
    for n in grid_neurons:
        for d in grid_dropout:
            model = dense_stacked(trainX, num_neuron=n, dropout=d)
            print("epochs={ep} : neurons={ne} : dropout={do}".format(ep=e, ne=n, do=d))
            for r in range(repeats):
                # split
                X_train_3x, X_test_3x, y_train_3x, y_test_3x = train_test_split(
                    df_shift_3.drop(["(t)date_ym", "(t)target"], axis=1),
                    df_shift_3["(t)target"],
                    test_size=0.15,
                    shuffle=True,
                    random_state=None,
                )
                # scale
                X_train_scaled_3x = scaler.fit_transform(X_train_3x)
                X_test_scaled_3x = scaler.transform(X_test_3x)
                X_scaled_3x = scaler.transform(X_3)
                # assign
                trainX = X_train_scaled_3
                y_train = y_train_3
                testX = X_test_scaled_3
                y_test = y_test_3
                devX = X_scaled_3
                y = y_3
                # fit
                # model = dense_stacked(trainX, num_neuron=n, dropout=d)
                lstm_fit(trainX, y_train, model, num_epoch=e)
                train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
                test_mape = model.evaluate(testX, y_test, verbose=0)[1]
                print(
                    "Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                        rep=r + 1, trainmape=train_mape, testmape=test_mape
                    )
                )
                result_end.loc[
                    r, "fix_train_" + str(e) + "eX" + str(n) + "nX" + str(d) + "d"
                ] = train_mape
                result_end.loc[
                    r, "fix_test_" + str(e) + "eX" + str(n) + "nX" + str(d) + "d"
                ] = test_mape
            K.clear_session()
print("DONE!")
# summarize results
result_end_describe = pd.DataFrame(result_end.describe())
print(result_end.describe().T)
result_end[result_end != 100].dropna().boxplot(figsize=(14, 7), rot=90)
print(result_end.describe().T.filter(like="test", axis=0))
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# (t-3)(t-2)(t-1): CONVOLUTIONAL DENSE model
# --------------------------------------------------------------------------------------------------
# Assign the train dataset
# reshape from [samples, timesteps] into [samples, timesteps, features]
trainX = X_train_scaled_3.reshape(
    X_train_scaled_3.shape[0], X_train_scaled_3.shape[1], 1
)
y_train = y_train_3
testX = X_test_scaled_3.reshape(X_test_scaled_3.shape[0], X_test_scaled_3.shape[1], 1)
y_test = y_test_3
devX = X_scaled_3.reshape(X_scaled_3.shape[0], X_scaled_3.shape[1], 1)
y = y_3

cnn_type = "Conv2Dense"
result_cnn_tune2 = pd.DataFrame()
grid_neurons = [300, 500]
grid_filters = [10, 30, 50, 70]
grid_kernel = [5, 10, 15]
grid_pool = [2, 3, 4, 5]
grid_dropout = [0.0, 0.1]
repeats = 30

for n in grid_neurons:
    for f in grid_filters:
        for k in grid_kernel:
            for p in grid_pool:
                for d in grid_dropout:
                    model = cnn_2dense(
                        trainX, num_neuron=n, num_filter=f, kernel=k, pool=p, dropout=d
                    )
                    print(
                        "neurons={ne} : filter={fi} : kernel={ke} : polling={po} : dropout={do}".format(
                            ne=n, fi=f, ke=k, po=p, do=d
                        )
                    )
                    for r in range(repeats):
                        lstm_fit(trainX, y_train, model)
                        train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
                        test_mape = model.evaluate(testX, y_test, verbose=0)[1]
                        print(
                            cnn_type
                            + " 3: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                                rep=r + 1, trainmape=train_mape, testmape=test_mape
                            )
                        )
                        result_cnn_tune2.loc[
                            r,
                            cnn_type
                            + "(t-3)train_"
                            + str(n)
                            + "nX"
                            + str(f)
                            + "fX"
                            + str(k)
                            + "kX"
                            + str(p)
                            + "pX"
                            + str(d)
                            + "d",
                        ] = train_mape
                        result_cnn_tune2.loc[
                            r,
                            cnn_type
                            + "(t-3)test_"
                            + str(n)
                            + "nX"
                            + str(f)
                            + "fX"
                            + str(k)
                            + "kX"
                            + str(p)
                            + "pX"
                            + str(d)
                            + "d",
                        ] = test_mape
                    K.clear_session()

print("DONE !")
# summarize results
result_cnn_tune = pd.DataFrame(result_cnn_tune.describe())
result_cnn_tune2[result_cnn_tune2 != 100].dropna().boxplot(figsize=(14, 7), rot=90)
print(result_cnn_tune2.describe().T)

result_cnn_tune2[result_cnn_tune2 != 100].dropna().filter(like="test", axis=1).boxplot(
    figsize=(14, 7), rot=90
)
print(result_cnn_tune2.describe().T.filter(like="test", axis=0).sort_values(by=["50%"]))
print(
    result_cnn_tune2.describe().T.filter(like="50fX10", axis=0).sort_values(by=["75%"])
)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# Define train, test and dev datasets
trainX = X_train_scaled_3d_4
y_train = y_train_4
testX = X_test_scaled_3d_4
y_test = y_test_4
devX = X_scaled_3d_4
y = y_4

# Experiment for numbers of EPOCHS and NEURONS
result_en = pd.DataFrame()
grid_epochs = [50, 100]  # 50
grid_neurons = [300, 500, 1000]  # 300
repeats = 50
for e in grid_epochs:
    for n in grid_neurons:
        print("epochs={ep} : neurons={ne}".format(ep=e, ne=n))
        for r in range(repeats):
            # Place the rigth (best) model
            model = lstm_bidirect(num_neuron=n)
            lstm_fit(model, num_epoch=e)
            train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
            test_mape = model.evaluate(testX, y_test, verbose=0)[1]
            print(
                "Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                    rep=r + 1, trainmape=train_mape, testmape=test_mape
                )
            )
            result_en.loc[r, "train_" + str(e) + "eX" + str(n) + "n"] = train_mape
            result_en.loc[r, "test_" + str(e) + "eX" + str(n) + "n"] = test_mape
            K.clear_session()
# summarize results
result_en_describe = pd.DataFrame(result_en.describe())
print(result_en.describe().T)
result_en[result_en != 100].dropna().boxplot(figsize=(14, 7), rot=90)
result_en.to_csv("result_en.csv")

# Experimen for DROPOUT ratio
result_drop = pd.DataFrame()
grid_drop = [0.1, 0.2, 0.3]  # 0.1
grid_recdrop = [0.1, 0.3, 0.5]  # 0.3 or 0.5
repeats = 40
for d in grid_drop:
    for rd in grid_recdrop:
        print("dropout={dp} : recurrent_dropout={rdp}".format(dp=d, rdp=rd))
        for r in range(repeats):
            # Place the rigth (best) model
            model = lstm_bidirect(
                num_neuron=500, alphalrelu=0.5, dropout=d, recurdropout=rd
            )
            lstm_fit(model, num_epoch=50)
            train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
            test_mape = model.evaluate(testX, y_test, verbose=0)[1]
            print(
                "Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                    rep=r + 1, trainmape=train_mape, testmape=test_mape
                )
            )
            result_drop.loc[
                r, "train_" + str(d) + "dropX" + str(rd) + "recdrop"
            ] = train_mape
            result_drop.loc[
                r, "test_" + str(d) + "dropX" + str(rd) + "recdrop"
            ] = test_mape
            K.clear_session()
# summarize results
result_drop_describe = pd.DataFrame(result_drop.describe())
print(result_drop.describe().T)
print(result_drop.describe().T.filter(like="test", axis=0))
result_drop[result_drop != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# result_drop.to_csv('result_drop.csv')

# Experimen for RECURRENT_DROPOUT ratio + nums of NEORONS & EPOCHS
result_den = pd.DataFrame()
grid_recdrop = [0.3, 0.5]
grid_epochs = [50, 100]
grid_neurons = [300, 500, 1000]
repeats = 40
for rd in grid_recdrop:
    for e in grid_epochs:
        for n in grid_neurons:
            print(
                "rec_dropout={rdp} : epochs={ep} : neurons={ne}".format(
                    rdp=rd, ep=e, ne=n
                )
            )
            for r in range(repeats):
                # Place the rigth (best) model
                model = lstm_bidirect(
                    num_neuron=n, alphalrelu=0.5, dropout=0.1, recurdropout=rd
                )
                lstm_fit(model, num_epoch=e)
                train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
                test_mape = model.evaluate(testX, y_test, verbose=0)[1]
                print(
                    "Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                        rep=r + 1, trainmape=train_mape, testmape=test_mape
                    )
                )
                result_den.loc[
                    r, "train_" + str(rd) + "recdrop_" + str(e) + "eX" + str(n) + "n"
                ] = train_mape
                result_den.loc[
                    r, "test_" + str(rd) + "recdrop_" + str(e) + "eX" + str(n) + "n"
                ] = test_mape
                K.clear_session()
# summarize results
result_den_describe = pd.DataFrame(result_den.describe())
print(result_den.describe().T)
print(result_den.describe().T.filter(like="test", axis=0))
result_den[result_den != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# result_den.to_csv('result_den.csv')


# Experiment for ACTIVATION in output layer
result_outact = pd.DataFrame()
grid_activ = ["linear", "sigmoid", "softsign", "hard_sigmoid"]
repeats = 50
for a in grid_activ:
    print("Dense activation:" + a)
    for r in range(repeats):
        # Place the rigth (best) model
        model = lstm_bidirect(
            num_neuron=500, alphalrelu=0.5, out_activ=a, dropout=0.1, recurdropout=0.3
        )
        lstm_fit(model, num_epoch=100)
        train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
        test_mape = model.evaluate(testX, y_test, verbose=0)[1]
        print(
            "Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                rep=r + 1, trainmape=train_mape, testmape=test_mape
            )
        )
        result_outact.loc[r, "train_" + a] = train_mape
        result_outact.loc[r, "test_" + a] = test_mape
        K.clear_session()
# summarize results
result_outact_describe = pd.DataFrame(result_outact.describe())
print(result_outact.describe().T)
print(result_outact.describe().T.filter(like="test", axis=0))
result_outact[result_outact != 100].dropna().boxplot(figsize=(14, 7), rot=90)
# result_den.to_csv('result_den.csv')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DTYING Relu investigation
# Define train, test and des datasets
trainX = X_train_scaled_3d_4
y_train = y_train_4
testX = X_test_scaled_3d_4
y_test = y_test_4
devX = X_scaled_3d_4
y = y_4


def lstm_vanilla_lrelu(
    num_neuron=500, activation="linear", alphalrelu=0.3, dropout=0, recurdropout=0
):
    # design VANILLA LSTM
    model = Sequential()
    model.add(
        LSTM(
            num_neuron,
            activation=activation,
            input_shape=(trainX.shape[1], trainX.shape[2]),
            dropout=dropout,
            recurrent_dropout=recurdropout,
        )
    )
    model.add(LeakyReLU(alpha=alphalrelu))  # add an advanced activation
    model.add(Dense(1))
    model.add(LeakyReLU(alpha=alphalrelu))  # add an advanced activation
    model.compile(loss="mae", optimizer="adam", metrics=["mape"])
    # model.summary()
    model.reset_states()
    return model


def lstm_vanilla_prelu(num_neuron=500, activation="linear", dropout=0, recurdropout=0):
    # design VANILLA LSTM
    model = Sequential()
    model.add(
        LSTM(
            num_neuron,
            activation=activation,
            input_shape=(trainX.shape[1], trainX.shape[2]),
            dropout=dropout,
            recurrent_dropout=recurdropout,
        )
    )
    model.add(PReLU())  # add an advanced activation
    model.add(Dense(1))
    model.add(PReLU())  # add an advanced activation
    model.compile(loss="mae", optimizer="adam", metrics=["mape"])
    # model.summary()
    model.reset_states()
    return model


result_relu = pd.DataFrame()
result_range = 40

grid_alpha = [0.7, 1]
# For LeakyReLU
for a in grid_alpha:
    for r in range(result_range):
        model = lstm_vanilla_lrelu(alphalrelu=a)
        lstm_fit(model)
        train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
        test_mape = model.evaluate(testX, y_test, verbose=0)[1]
        print(
            "LeakyReLU: alpha{al}: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
                al=a, rep=r + 1, trainmape=train_mape, testmape=test_mape
            )
        )
        result_relu.loc[r, "LeakyReLU_" + str(a) + "_train_mape"] = train_mape
        result_relu.loc[r, "LeakyReLU_" + str(a) + "_test_mape"] = test_mape

# For RReLU
for r in range(result_range):
    model = lstm_vanilla_prelu()
    lstm_fit(model)
    train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
    test_mape = model.evaluate(testX, y_test, verbose=0)[1]
    print(
        "ParamReLU: Repeat={rep}: TrainMAPE={trainmape} TestMAPE={testmape}".format(
            rep=r + 1, trainmape=train_mape, testmape=test_mape
        )
    )
    result_relu.loc[r, "PReLU_train_mape"] = train_mape
    result_relu.loc[r, "PReLU_test_mape"] = test_mape

result_relu_describe = pd.DataFrame(result_relu.describe())
print(result_relu.describe().T)
result_relu.boxplot(figsize=(14, 7), rot=90)
print(result_relu.describe().T.filter(like="test", axis=0))
result_relu.filter(like="test", axis=1).boxplot(figsize=(14, 7), rot=90)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VIEW BEST MODEL THROUGTH MULTIPLE RANDOM SPLIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from LSTM_models import cnn_2denseX

result_randsplitX = pd.DataFrame()
result_range = 50
cnn_type = "Conv2Dense"
for e in [50, 200]:
    for n in [500, 1000]:
        for f in [20, 50, 100]:
            for k in [10, 20]:
                for p in [3, 4, 5]:
                    for d in [0.0, 0.2]:
                        # model = cnn_2denseX(num_neuron=n, num_filter=50, kernel=10, pool=4, dropout=d)
                        #            print('Epochs={ep} : neurons={ne} : alphaReLU={al}'
                        #                  .format(ep=e, ne=n, al=a))
                        for r in range(result_range):
                            # split
                            (
                                X_train_3x,
                                X_test_3x,
                                y_train_3x,
                                y_test_3x,
                            ) = train_test_split(
                                df_shift_3.drop(["(t)date_ym", "(t)target"], axis=1),
                                df_shift_3["(t)target"],
                                test_size=0.15,
                                shuffle=True,
                                random_state=None,
                            )
                            # scale
                            X_train_scaled_3x = scaler.fit_transform(X_train_3x)
                            X_test_scaled_3x = scaler.transform(X_test_3x)
                            X_scaled_3 = scaler.transform(X_3)
                            # assign
                            trainX = X_train_scaled_3x.reshape(
                                X_train_scaled_3x.shape[0],
                                X_train_scaled_3x.shape[1],
                                1,
                            )
                            y_train = y_train_3x
                            testX = X_test_scaled_3x.reshape(
                                X_test_scaled_3x.shape[0], X_test_scaled_3x.shape[1], 1
                            )
                            y_test = y_test_3x
                            devX = X_scaled_3.reshape(
                                X_scaled_3.shape[0], X_scaled_3.shape[1], 1
                            )
                            y = y_3
                            # fit
                            model = cnn_2denseX(
                                num_neuron=n, num_filter=f, kernel=k, pool=p, dropout=d
                            )
                            history = lstm_fit(
                                trainX, y_train, model, num_epoch=e, val_split=0.15
                            )
                            train_mape = model.evaluate(trainX, y_train, verbose=0)[1]
                            test_mape = model.evaluate(testX, y_test, verbose=0)[1]
                            print(
                                "Epochs={ep} : neurons={ne} : filter={fl} : kernel={ke} : pool={po}: drop={dr}: Repeat={rep} \n"
                                "TrainMAPE={trainmape} TestMAPE={testmape}".format(
                                    ep=e,
                                    ne=n,
                                    fl=f,
                                    ke=k,
                                    po=p,
                                    dr=d,
                                    rep=r + 1,
                                    trainmape=train_mape,
                                    testmape=test_mape,
                                )
                            )
                            step = (
                                str(e)
                                + "e_"
                                + str(n)
                                + "n_"
                                + str(f)
                                + "f_"
                                + str(k)
                                + "k_"
                                + str(p)
                                + "p"
                                + str(d)
                                + "d"
                            )
                            result_randsplitX.loc[
                                r, cnn_type + "(t3)train_" + step
                            ] = train_mape
                            result_randsplitX.loc[
                                r, cnn_type + "(t3)test_" + step
                            ] = test_mape
                            result_randsplitX.loc[r, cnn_type + "(t3)ratio_" + step] = (
                                test_mape / train_mape
                            )
                            # predict
                            pred, score_train, score_test, score_dev = calcPredict(
                                df, model, trainX, y_train, testX, y_test, devX, y
                            )
                            # save
                            model_randsplit = [
                                model,
                                history,
                                pred,
                                score_train,
                                score_test,
                                score_dev,
                                cnn_type,
                            ]
                            # view
                            # viewModelMAPE(model_randsplit, figsize=(14,4))
                            viewPredict(model_randsplit, figsize=(14, 4))
                            K.clear_session()
print("========= VALIO!! ===========")
# result
result_randsplitX.to_csv("result_randsplitX_inLopp.csv")
result_randsplitX.filter(like="test", axis=1).boxplot(figsize=(28, 7), rot=90)

print(result_randsplitX.describe().T, "Results for RANDOM SPLIT in Loop")
print(
    result_randsplitX.describe().T.filter(like="test", axis=0).sort_values(by=["50%"]),
    "Results for RANDOM SPLIT in loop",
)
print(
    result_randsplitX.describe()
    .T.filter(like="ratio", axis=0)
    .sort_values(by=["mean"]),
    "Results for RANDOM SPLIT in loop",
)

print(
    result_randsplitX.describe().T.filter(like="test", axis=0).sort_values(by=["min"])
)

result_randsplitX[
    [
        "Conv2Dense(t-3)train_200e_1000n_25f_20k_5p",
        "Conv2Dense(t-3)test_200e_1000n_25f_20k_5p",
    ]
].plot.line()
result_randsplitX.filter(like="test_200e_500n_50f", axis=1).plot.line()

# --------------------------------------------------------------------------------------------------
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VIEW BEST MODEL THROUGTH MULTIPLE RANDOM SPLIT AND CROSS-VALIDATION
ITERATED K-FOLDVALIDATIONWITHSHUFFLING
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from LSTM_models import cnn_2denseX

model = lstm_bidirect_stacked(trainX)
from keras.utils import plot_model

plot_model(model, to_file="lstm_bidirect_stacked.png")

#################################### E N D #########################################################
