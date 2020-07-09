# -*- coding: utf-8 -*-
"""
Created on Jan 15 2020
    This code takes the 'dflow' dataset
    and split it on train/validation/test sets
    and get the parameters for scaling the numeric variables
    and for embedding the categorical
@author: mikhail.galkin
"""
#%% [markdown] -----------------------------------------------------------------
### Start running code
##### _ '----' in cell's title means printing snipet and may be non-mandatory for real_
##### _ '====' in cell's title means saving snipet and may be non-mandatory for real_
#%% Importing required packages
from __future__ import absolute_import, division, print_function, unicode_literals
import time

start_time = time.time()
import os
import json
import pickle
import math
import pandas as pd

print("pandas ver.:", pd.__version__)
import numpy as np

print("numpy ver.:", np.__version__)
import matplotlib.pyplot as plt

print("matplotlib ver.:", matplotlib.__version__)
from tensorflow import feature_column

#%% Load data
# get path
f = os.getcwd().replace(os.path.basename(os.getcwd()), "")
# load data
start_time_load = time.time()
df = pd.read_csv(
    f + "data_in/tf/dflow_20_loans.csv",
    sep=",",
    decimal=".",
    header=0,
    parse_dates=["first_status_day_date"],
    date_parser=lambda col: pd.to_datetime(col).strftime("%Y-%m-%d"),
    infer_datetime_format=True,
    low_memory=False,
    float_precision="round_trip",
)
# ------------------------------------------------------------------------------
# print(list(df))
print("Data loaded for --- %s seconds ---" % (time.time() - start_time_load))

#%% [markdown] -----------------------------------------------------------------
#### INFO block
#%% INFO #1: -------------------------------------------------------------------
print(df.info(), "\n")
print("#NA =", df.isna().sum().sum())
print(df.dtypes.value_counts())
#%% INFO#2: Number of applications: --------------------------------------------
apps_count = (
    df.groupby([df.first_status_day_date.dt.to_period("M"), "customer_type"])
    .size()
    .unstack()
    .assign(all=df.groupby(df.first_status_day_date.dt.to_period("M")).size())
)
print(apps_count)
apps_count.plot()
del apps_count
#%% INFO#3: --------------------------------------------------------------------
print(df.groupby(df["first_status_day_date"].dt.to_period("M")).size())
print(
    df.groupby(df["first_status_day_date"].dt.to_period("M")).size().cumsum()
    / len(df.index)
)

#%% Create lists of columns names
cols_inf = [
    "id",
    "first_status_day_date",
    "customer_type",
    "customer_id",
    "act_profit",
    "act_loan_numinstal",
    "act_loan_amount",
    "act_dpd",
]
cols_y = ["yn", "yd", "y3", "y2", "y1", "y0_class", "y0_2_1", "y0_2", "y0_1", "y0"]
cols_x = [x for x in list(df) if x not in (cols_inf + cols_y)]
# ------------------------------------------------------------------------------
print("cols_x:", len(cols_x), "\n", cols_x)

#%% Get dictionary of columns' types
# ------------------------------------------------------------------------------
print(df.get_dtype_counts())

# Categorical features
## to indicator cols
cols_x_indi = [x for x in cols_x if "act_app_gender" in x]
## to embedding cols
cols_x_embe = [
    x
    for x in list(df[cols_x].select_dtypes(include=["object"]))
    if x not in cols_x_indi
]

# Numeric features
## w\o normalizing
binary = [
    "act_changed_city",
    "act_changed_code_zip",
    "act_changed_email_domain",
    "act_changed_iban_bank",
    "act_is_name_first_in_email",
    "act_is_name_last_in_email",
    "act_is_part_first_name_in_email",
    "act_is_part_last_name_in_email",
    "last_sold",
    "last_late_00",
    "last_late_03",
    "last_late_07",
    "last_late_15",
    "last_late_30",
    "last_late_60",
    "last_ontime_00",
    "last_ontime_03",
    "last_ontime_07",
    "last_ontime_15",
    "last_ontime_30",
    "last_ontime_60",
    "prev_dummy_late_00",
    "prev_dummy_late_03",
    "prev_dummy_late_07",
    "prev_dummy_late_15",
    "prev_dummy_late_30",
    "prev_dummy_late_60",
    "prev_dummy_06m_late_00",
    "prev_dummy_12m_late_00",
    "prev_dummy_24m_late_00",
]
cols_x_bina = [x for x in cols_x for y in binary if y in x]
del binary
## with normalizing
cols_x_nume = [
    x
    for x in list(df[cols_x].select_dtypes(include=["float64", "int64"]))
    if x not in cols_x_bina
]

#%% print ----------------------------------------------------------------------
print(
    "\n len cols_x_indi:",
    len(cols_x_indi),
    "\n",
    "len cols_x_embe:",
    len(cols_x_embe),
    "\n",
    "len cols_x_bina:",
    len(cols_x_bina),
    "\n",
    "len cols_x_nume:",
    len(cols_x_nume),
    "\n",
    "? Equal for len(cols_x):",
    len(cols_x),
    len(cols_x) == len(cols_x_indi + cols_x_embe + cols_x_bina + cols_x_nume),
)
#%% Create dictionary of columns
dic_cols = {
    "y": cols_y,
    "x": cols_x,
    "x_bina": cols_x_bina,
    "x_indi": cols_x_indi,
    "x_nume": cols_x_nume,
    "x_embe": cols_x_embe,
    "inf": cols_inf,
}

#%% Split the dataframe into train, validation and test
## The dataset was a single CSV file. We'll split this into train, validation, and test sets
data_test = "2019-07-01"
data_val = "2019-04-01"
df_test = df[df["first_status_day_date"] >= data_test]
df_train = df[df["first_status_day_date"] < data_val]
df_val = df[
    (df["first_status_day_date"] < data_test)
    & (df["first_status_day_date"] >= data_val)
]

#%% print ----------------------------------------------------------------------
len_df = len(df)
print(
    len(df_test),
    "test examples",
    round(len(df_test) / len_df * 100, 2),
    "% of whole data \n",
    df_test.groupby(["y0_1"])["y0_1"].size().apply(lambda x: x / len(df_test)),
    "\n",
)
print(
    len(df_train),
    "train examples",
    round(len(df_train) / len_df * 100, 2),
    "% of whole data \n",
    df_train.groupby(["y0_1"])["y0_1"].size().apply(lambda x: x / len(df_train)),
    "\n",
)
print(
    len(df_val),
    "validation examples",
    round(len(df_val) / len_df * 100, 2),
    "% of whole data \n",
    df_val.groupby(["y0_1"])["y0_1"].size().apply(lambda x: x / len(df_val)),
    "\n",
)
print("Whole data:")
df.groupby(["y0_1"])["y0_1"].size().apply(lambda x: x / len_df)

#%% REMOVE df
del df

#%% [markdown]------------------------------------------------------------------
#### Get the params and method for normalizing numeric data
#%% Function to get normalization parameters
def get_norm_params(df_train, cols):
    """Get the normalization parameters (E.g., mean, std) for df_train for
    features. We will use these parameters for training, eval, and predictions."""

    def params(col):
        mini = df_train[col].min()
        maxi = df_train[col].max()
        mean = df_train[col].mean()
        std = df_train[col].std()
        q05 = df_train[col].quantile(q=0.05)
        q10 = df_train[col].quantile(q=0.10)
        q20 = df_train[col].quantile(q=0.20)
        q25 = df_train[col].quantile(q=0.25)
        q50 = df_train[col].quantile(q=0.50)
        q75 = df_train[col].quantile(q=0.75)
        q80 = df_train[col].quantile(q=0.80)
        q90 = df_train[col].quantile(q=0.90)
        q95 = df_train[col].quantile(q=0.95)
        return {
            "min": mini,
            "max": maxi,
            "mean": mean,
            "std": std,
            "q05": q05,
            "q10": q10,
            "q20": q20,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q80": q80,
            "q90": q90,
            "q95": q95,
        }

    norm_params = {}
    for col in cols:
        col_00 = col.replace(col[0:3], "00_")
        norm_params[col] = params(col_00)

    df_norm_params = pd.DataFrame.from_dict(norm_params)
    return df_norm_params


#%% Function to get embedding parameters
def get_embe_params(df_train, cols):
    import math

    def dims(col):
        vocab = ["0"]
        if col[3:] == "y0_class":
            dim = 2
            nunique = 6
            vocab = list(df_train["y0_class"].unique())
        else:
            col_00 = "00_" + col[3:]
            dim = math.ceil(df_train[col_00].nunique() ** 0.25)
            nunique = df_train[col].nunique()
            vocab = vocab + list(df_train[col_00].unique())

        return {"dim": dim, "nunique": nunique, "vocab": vocab}

    embe_params = {}
    for col in cols:
        embe_params[col] = dims(col)
    # return embe_params
    # df_embe_params = pd.DataFrame.from_dict(embe_params)
    return embe_params


#%% Get the normalization parameters for numeric columns
dflow_norm_params = get_norm_params(df_train, cols_x_nume)
# view result -----------------------------------------------------------------
dflow_norm_params

#%% Get the embedding parameters for categorical columns
dic_embe_params = get_embe_params(df_train, cols_x_embe)
# view result -----------------------------------------------------------------
dic_embe_params

#%% [markdown] -----------------------------------------------------------------
#### Choose and create features columns we will use
feature_cols = []
#%% Get TF feature columns
for col in dic_cols["x"]:
    if col in dic_cols["x_nume"]:
        # print(col, 'Numeric')
        feature_cols.append(feature_column.numeric_column(col))
    elif col in dic_cols["x_bina"]:
        # print(col, 'Binary')
        feature_cols.append(feature_column.numeric_column(col))
    elif col in dic_cols["x_indi"]:
        # print(col, 'Indicator')
        gender = feature_column.categorical_column_with_vocabulary_list(col, ["m", "f"])
        gender_ohe = feature_column.indicator_column(gender)
        feature_cols.append(gender_ohe)
    elif col in dic_cols["x_embe"]:
        # print(col, 'Embedding')
        vocabulary_list = dic_embe_params[col]["vocab"]
        dimension = dic_embe_params[col]["dim"]
        categorical = feature_column.categorical_column_with_vocabulary_list(
            col, vocabulary_list=vocabulary_list
        )
        embedding = feature_column.embedding_column(categorical, dimension=dimension)
        feature_cols.append(embedding)

del (col, gender, gender_ohe, vocabulary_list, dimension, categorical, embedding)
#%% View features columns ------------------------------------------------------
print(len(feature_cols))
print(*feature_cols, sep="\n")

#%% Save feature_columns information in file ===================================
with open(f + "data_in/tf/tf_feature_cols.txt", "wb") as fp:
    pickle.dump(feature_cols, fp)

#%% Save dictionary of cols ====================================================
with open(f + "data_in/tf/dic_cols.json", "w") as fp:
    json.dump(dic_cols, fp)
#%% Save normalization parameters ==============================================
# as data frame
dflow_norm_params.to_csv(
    f + "data_in/tf/dflow_norm_params.csv",
    index=True,
    sep=",",
    decimal=".",
    header=True,
    encoding="utf-8",
)
#%% Save embedding parameters ==================================================
# as json dictionary
with open(f + "data_in/tf/dic_embe_params.json", "w") as fp:
    json.dump(dic_embe_params, fp)
#%% Save datasets ==============================================================
start_time_save = time.time()
df_test.to_csv(
    f + "data_in/tf/dflow_test.csv", index=False, sep=",", decimal=".", header=True
)
df_val.to_csv(
    f + "data_in/tf/dflow_val.csv", index=False, sep=",", decimal=".", header=True
)
df_train.to_csv(
    f + "data_in/tf/dflow_train.csv", index=False, sep=",", decimal=".", header=True
)
# ------------------------------------------------------------------------------
print(
    "DONE! Data splitted and params saved for --- %s seconds ---"
    % (time.time() - start_time)
)

#%% Choosing a normalizing method of the numeric data --------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

col = "00_prev_total_profit"
print("Not scaled", "\n", df_train["00_prev_total_profit"].iloc[97:111])

# Standard scaling manual
df_scaled_std = df_train[cols_x_nume] - dflow_norm_params.loc["mean", :]
df_scaled_std /= dflow_norm_params.loc["std", :]
print("Standard", "\n", df_scaled_std["00_prev_total_profit"].iloc[97:111])
print(
    "Min afrer Standard:",
    df_scaled_std["00_prev_total_profit"].min(),
    "\n",
    "Max afrer Standard:",
    df_scaled_std["00_prev_total_profit"].max(),
)
# Standard scaling from Sklearn
scaler = StandardScaler()
df_std = scaler.fit_transform(df_train[[col]].to_numpy())
print("mean:", scaler.mean_, "sdt:", scaler.var_)
print(df_std[97:111, 0])

# Robust scaling manual
df_scaled_rbs = df_train[cols_x_nume] - dflow_norm_params.loc["q50", :]
df_scaled_rbs /= dflow_norm_params.loc["q75", :] - dflow_norm_params.loc["q25", :]
print("Robust 75-25", "\n", df_scaled_rbs["00_prev_total_profit"].iloc[97:111])
print(
    "Min afrer Robust:",
    df_scaled_rbs["00_prev_total_profit"].min(),
    "\n",
    "Max afrer Robust:",
    df_scaled_rbs["00_prev_total_profit"].max(),
)
# Robust scaling from Sklearn
scaler = RobustScaler()
df_robust = scaler.fit_transform(df_train[[col]].to_numpy())
print("q50:", scaler.center_, "q75-q25:", scaler.scale_)
print(df_robust[97:111, 0])

# MinMax scaling manual
df_scaled_mmax = df_train[cols_x_nume] - dflow_norm_params.loc["min", :]
df_scaled_mmax /= dflow_norm_params.loc["max", :] - dflow_norm_params.loc["min", :]
print("Min-Max", "\n", df_scaled_mmax["00_prev_total_profit"].iloc[97:111])
print(
    "Min afrer MinMax:",
    df_scaled_mmax["00_prev_total_profit"].min(),
    "\n",
    "Max afrer MinMax:",
    df_scaled_mmax["00_prev_total_profit"].max(),
)
# MinMax scaling from Sklearn
scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(df_train[[col]].to_numpy())
print("min:", scaler.data_min_, "max:", scaler.data_max_)
print(df_minmax[97:111, 0])

# View on plots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
ax1.set_title("Before Scaling")
sns.kdeplot(df_train[col], ax=ax1)
ax2.set_title("After Scaling")
sns.kdeplot(df_scaled_std[col], ax=ax2, label="Standard")
sns.kdeplot(df_scaled_rbs[col], ax=ax2, label="Robust")
sns.kdeplot(df_scaled_mmax[col], ax=ax2, label="Min-Max")
plt.show()

del (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    scaler,
    col,
    ax1,
    ax2,
    fig,
    df_scaled_std,
    df_std,
    df_scaled_rbs,
    df_robust,
    df_scaled_mmax,
    df_minmax,
)

################################## E N D #######################################
