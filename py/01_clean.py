# -*- coding: utf-8 -*-
"""
Created on Dec 02 2019
    This code takes the [data_raw] raw dataset after SQL
       and makes data cleaning,
       creates the target variables
       and set the columns order
@author: mikhail.galkin
"""
#%% [markdown] -----------------------------------------------------------------
## To Do
### Cleaning, mapping data and handling missing values:
#### 1. [X] Load data:
#      1.1 [X] rename targets
#      1.2 [X] drop rows with target==2 (opened loans)
#### 2. [X] Starting handling:
#      2.1 [X] column's names to lowercase
#      2.2 [X] strings to lowercase
#      2.3 [X] 'true'\'false' to 1\0
#      2.4 [X] trim whitespaces
#### 3. [X] Transform potential leakage data from 'act_' to 'last_':
#      3.1 [X] 'act_sold'                 -> 'last_sold'
#      3.2 [X] 'act_loan_expenses'        -> 'last_loan_expenses'
#      3.3 [X] 'act_loan_interest'        -> 'last_loan_interest'
#      3.4 [X] 'act_loan_term'            -> 'last_loan_term'
#      3.5 [X] 'act_loan_term_real'       -> 'last_loan_term_real'
#### 4. [X] Missing values:
#      4.1 [X] NA for 'target_2'.fillna(-1)
#      4.2 [X] NA for Credit Bureau scores
#      4.3 [X] missing categorical        -> '_missing'
#      4.4 [X] missing numerical          -> 0
### Targets:
#      0.. [X] create new artifical 'y0_2_1' target
#### 1. [X] For requested loan:
#      1.1 [X] y0_1: result on 1st instalment:
#             * binary-class: (Sigmoid):
#                    ** 'bad'      :=0
#                    ** 'good'     :=1
#      1.2 [X] y0_2: result on 2nd instalment:
#             * triple-class: (Softmax\Than):
#                    ** not used   :=-1
#                    ** 'bad'      :=0
#                    ** 'good'     :=1
#      1.3 [X] y0: result of whole requested loan:
#             * binary-class: (Sigmoid):
#                    ** 'bad'      :=0
#                    ** 'good'     :=1
#      1.4 [X] y0_class: Final class of requested loan:
#             * 6x-multi-class : (Softmax): string
#### 2. [X] For future loans:
#      2.1 [X] yd: Loan's number which was(will) defaulted:
#             * numeric: (? Softmax)): #
#                    ** loans number is unknown  := -1
#                    ** loans number is known    := interger
#      2.2 [X] y1, y2, y3: PD of loan(+1), (+2), (+3):
#             * triple-class: (Softmax):
#                    ** exactly will be 'bad'    := 0
#                    ** exactly will be 'good'   := 1
#                    ** unknown                  := -1
#      2.3 [X] yn: Rest of numbers of loans before default:
#             * numeric: (Softmax): #
#                    ** unknown (last loan is 'good')   := -1
#                    ** actual loan is 'bad'            := 0
#                    ** number of loans before default  := interger


#%% Load libraries for timing
import time
start_time = time.time()
#%% Importing required packages
import os
import pandas as pd ;print('pandas ver.:', pd.__version__)
import numpy as np ;print('numpy ver.:', np.__version__)
import matplotlib ;print('matplotloib ver.:', matplotlib.__version__)
import matplotlib.pyplot as plt
import seaborn as sns ;print('seaborn ver.:', sns.__version__)
import re

#%% For reflecting graphics in IPython\notebooks window correctly
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# Set up needed options
pd.set_option('display.max_columns', 18)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#%% Set up parameters for Matplotlib output
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 5
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = '--'

#%% [markdown] -----------------------------------------------------------------
#### Load data
#%% Load data
# get path
f = os.getcwd().replace(os.path.basename(os.getcwd()), '')
# load data
df = pd.read_csv(f + 'data_in/data_raw.csv', sep=',', decimal='.', header=0,
                 dtype={
                     'act_iban_bank': 'int64'
                     , 'act_code_zip2': 'int64'
                     , 'act_arrived_hour': 'int64'
                     , 'act_arrived_day': 'int64'
                     , 'act_arrived_weekday': 'int64'
                      'cb_score_class': 'str'
                 },
                 parse_dates=[
                     'first_status_day_date'
                 ],
                 date_parser=lambda col: pd.to_datetime(col).strftime('%Y-%m-%d')
                 , infer_datetime_format=True
                 )
df.iloc[:, :10].head() # review

#%% INFO -----------------------------------------------------------------------
print(df.info(), '\n')
print('#NA =', df.isna().sum().sum())
print(df.dtypes.value_counts())

#%% Convert dtypaes for Pandas#1.0
# df = df.convert_dtypes()
# print('#NA =', df.isna().sum().sum())
# print(df.dtypes.value_counts())

#%% [markdown] -----------------------------------------------------------------
#### Rename targets and drop opened loans
#%% Rename target
df.rename(columns={'target_1st': 'y0_1',
                   'target_2nd': 'y0_2'}
                   , inplace=True
          )
pd.crosstab(df.y0_1.fillna(9), df.y0_2.fillna(9), margins=True)

#%% Fill NA for 'target_2nd'
# customer_id=2598448 # for checking
df['y0_2'] = df['y0_2'].fillna(-1)
df['y0_2'] = df['y0_2'].astype('int64')

#%% Drop opened loans with [targets] = 2
# INFO: number of applications before dropping
apps_bef = df.groupby(df.first_status_day_date.dt.to_period("M")).size()

# drop the rows
df = df.drop(df[(df.y0_1==2) | (df.y0_2==2)].index)

# INFO: number of application after dropping
apps_aft = df.groupby(df.first_status_day_date.dt.to_period("M")).size()
# INFO: view resutls -----------------------------------------------------------
apps_bef.to_frame(name='before').join(apps_aft.to_frame(name='after'))\
       .plot(title='#apllications after deleting [targets]==2')
del(apps_bef, apps_aft)

#%% [markdown] -----------------------------------------------------------------
#### Starting primitive handling
#%% Handling
# Map column names to all str values lowercase
df.columns = [x.lower() for x in df.columns]
# To lowercase all str values and delete whitespaces
df = df.apply(lambda x: x.str.lower().str.strip() if x.dtype==object else x)
# Replace true\false on 1\0
df = df.apply(lambda x: x*1 if x.dtype==bool else x)
# Detele whitespaces
df = df.apply(lambda x: x.str.strip() if x.dtype==object else x)

#%% [markdown] -----------------------------------------------------------------
#### Shift the act_ columns to the last_
#%% Shift the 'act_sold'
df['last_sold'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['act_sold'].shift(1).fillna(0).astype('int64')

#%% Shift the 'act_loan_expenses'
df['last_loan_expenses'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['act_loan_expenses'].shift(1).fillna(0)

#%% Shift the 'act_loan_interest'
df['last_loan_interest'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['act_loan_interest'].shift(1).fillna(0)

#%% Shift the 'act_loan_term'
df['last_loan_term'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['act_loan_term'].shift(1).fillna(0)\
              .astype('int64')

#%% Shift the 'act_loan_term_real'
df['last_loan_term_real'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['act_loan_term_real'].shift(1).fillna(0)\
              .astype('int64')

#%% Check: View all customer & their loans which were sold at least one time
# INFO
df.loc[df.customer_id.isin(
              ['2600185']
              #df.loc[df.act_sold == 1, 'customer_id']
              ),
              ['customer_id', 'act_loan_number', 'act_sold', 'last_sold',
              'act_loan_expenses', 'last_loan_expenses',
              'act_loan_interest', 'last_loan_interest',
              'act_loan_term_real', 'last_loan_term_real',
              'act_loan_term', 'last_loan_term']]\
       .sort_values(['customer_id', 'act_loan_number'])

#%% [markdown] -----------------------------------------------------------------
#### Treat missing values
#%% INFO
# cb_bv_score\ cb__riskrate\ cb__score\ cb_score_class
print('\nMean values\n',
df[['cb_bv_score', 'cb__riskrate', 'cb__score']]\
       [df['customer_type']=='nc'].agg(['mean'])
)
print('\nMissing in new customer\n',
df[['cb_bv_score', 'cb__riskrate', 'cb__score', 'cb_score_class']]\
       [df['customer_type']=='nc'].isnull().sum()
)
print('\nAll missing\n',
df[['cb_bv_score', 'cb__riskrate', 'cb__score', 'cb_score_class']]\
       .isnull().sum()
)

#%% Fill NA for CB info
# Boniversum Score - by mean
# df['cb_bv_score'].fillna(0, inplace=True)
df.loc[df['customer_type']=='nc', 'cb_bv_score']=\
       df.loc[df['customer_type']=='nc', 'cb_bv_score']\
              .fillna(df['cb_bv_score'].mean())
# Schufa Risk Rate - by mean
df.loc[df['customer_type']=='nc', 'cb__riskrate']=\
       df.loc[df['customer_type']=='nc', 'cb__riskrate']\
              .fillna(df['cb__riskrate'].mean())
# Schufa Score - by mean
df.loc[df['customer_type']=='nc', 'cb__score']=\
       df.loc[df['customer_type']=='nc', 'cb__score']\
              .fillna(df['cb__score'].mean())

# Get min-max range of Score for Schufa score
schufa_score_class = \
df[df['customer_type']=='nc'].groupby(['cb_score_class'])['cb__score'].agg(['min', 'max', 'count'])
schufa_score_class
# Put in a Schufa's score class
df.loc[df['customer_type']=='nc', ['cb_score_class']] =\
df.loc[df['customer_type']=='nc', ['cb_score_class']]\
       .fillna(
              schufa_score_class[
                     (schufa_score_class['min']<df['cb__score'].mean())
                     & (schufa_score_class['max']>=df['cb__score'].mean())
                     ].index.values[0]
)

#%% Get dictionary of columns' types
cols_types = df.dtypes.groupby(df.dtypes).groups
cols_types = {k.name: v for k, v in cols_types.items()}
 # view types ------------------------------------------------------------------
[*cols_types]

# %% Treat missing values in categorical
# df[cols_types['object']].isna().sum() # amount of NA's
for col in cols_types['object']:
       #sub = '_'+col
       sub = '_missing'
       df[col] = df[col].fillna(sub)
# ------------------------------------------------------------------------------
df.groupby('cb_score_class').size()
df.groupby('act_ga_browser').size()
# %% Treat missing values in numerics
df.isna().sum().sum() # amount of NA's
for col in cols_types['int32']:
       df[col] = df[col].fillna(0)
       df[col] = df[col].astype('int64')

for col in cols_types['int64']:
       df[col] = df[col].fillna(0)

for col in cols_types['float64']:
       df[col] = df[col].fillna(0)
del(sub, col)

#%% [markdown] -----------------------------------------------------------------
#### Handling targets
#%% Create new 'y0_2_1' target
# INFO
print('\nBefore\n',
       pd.crosstab([df.act_loan_numinstal, df.y0_1.fillna(-999)], df.y0_2.fillna(-999)))
# Create new y0_2 target
df['y0_2_1'] = df['y0_2']
df.loc[(df['y0_2']==-1) & (df['y0_1']==0), 'y0_2_1'] = 0
# INFO
print('\nAfter\n',
       pd.crosstab([df.act_loan_numinstal, df.y0_1], df.y0_2_1))

#%% Define the function for current loan result (target): y0
# y0: PD of reqiested\actual loan
# y0_class: Final class of requested\actual loan
# yd: Number of loan defaulted
def loan0_result(df):
       loan_number = df['act_loan_number']
       loan_numinstal = df['act_loan_numinstal']
       y0_1 = df['y0_1']
       y0_2 = df['y0_2']
       y0, y0_class, yd = 0, 'o', 0
       if loan_numinstal == 1:
              if y0_1 == 1:
                    y0, y0_class, yd = 1, 'i-', -1
              elif y0_1 == 0:
                    y0, y0_class, yd = 0, 'o-', loan_number
       elif loan_numinstal >= 2:
              if y0_1 == 1:
                     if y0_2 == 1:
                           y0, y0_class, yd = 1, 'ii', -1
                     elif y0_2 == 0:
                           y0, y0_class, yd = 0, 'io', loan_number
              elif y0_1 == 0:
                     if y0_2 == 1:
                            y0, y0_class, yd = 1, 'oi', -1
                     else:
                            y0, y0_class, yd = 0, 'oo', loan_number
       return pd.Series([y0, y0_class, yd], ['y0', 'y0_class', 'yd'])

#%% Apply function to current loan
df[['y0', 'y0_class', 'yd']] = df[['act_loan_number', 'act_loan_numinstal',
                                   'y0_1', 'y0_2']]\
    .apply(loan0_result, axis=1)
# Explore result
df.groupby('y0_class').size()

#%% Fill gaps the loan's number which will be defaulted
df['yd'] = df.sort_values(['customer_id', 'act_loan_number'])\
    .groupby(by=['customer_id'])['yd']\
    .apply(lambda x: x.replace(to_replace=-1, method='bfill'))
# Explore 'yd'
df.groupby('yd').size()
df.yd.hist(bins=df.yd.max())
sns.distplot(df.yd[df.yd!=0], kde=False, bins=df.yd.max())

#%% Create targets for PD of (+1), (+2), (+3) loans
df['y1'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['y0'].shift(-1).fillna(-1)
df['y1'] = df['y1'].astype('int64')

df['y2'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['y0'].shift(-2).fillna(-1)
df['y2'] = df['y2'].astype('int64')

df['y3'] = df.sort_values(['customer_id', 'act_loan_number'])\
       .groupby(by=['customer_id'])['y0'].shift(-3).fillna(-1)
df['y3'] = df['y3'].astype('int64')

#%% Rest of numbers of loans before default
df['yn'] = [-1 if x<0 else x for x in (df['yd']-df['act_loan_number'])]
df.yn.hist(bins=len(range(df.yn.min(), df.yn.max(), 1)))

#%% [markdown] -----------------------------------------------------------------
#### Handling redundant columns and rows
#%% Dicionary of lists with information about columns
# Creating an empty dictionary
cols_info = {}
# Adding lists as value
cols_info['del'] = ['id_ec', 'id_nc', 'id_origin_guid', 'id_source'
                     , 'act_sold', 'act_loan_expenses'
                     , 'act_loan_interest', 'act_loan_term_real', 'act_loan_term'
                     , 'last_loan_expenses'
                     , 'last_score', 'last_score_amount', 'last_score_class']

cols_info['inf'] = ['id', 'customer_id', 'customer_type'
                     , 'first_status_day_date'
                     , 'act_loan_amount', 'act_loan_numinstal'
                     , 'act_dpd', 'act_profit']

cols_info['id'] = [x for x in df.columns if re.search('^id|_id', x)]
cols_info['act'] = [x for x in df.columns if re.search('^act_', x)]
cols_info['cb'] = [x for x in df.columns if re.search('^cb_', x)]
cols_info['last'] = [x for x in df.columns if re.search('^last_', x)]
cols_info['prev'] = [x for x in df.columns if re.search('^prev_', x)]
cols_info['target'] = [x for x in df.columns if re.search('^y', x)]

#%% Drop needless and separate redundant columns
df = df.drop(cols_info['del'], axis=1)

#%% # Remove zero-variance features # added 2020.03.02
cols_inf = ['id', 'first_status_day_date', 'customer_type', 'customer_id',
    'act_profit', 'act_loan_numinstal', 'act_loan_amount', 'act_dpd']
cols_y = ['yn', 'yd', 'y3', 'y2', 'y1', 'y0_class', 'y0_2_1', 'y0_2', 'y0_1', 'y0']
cols_x = [x for x in list(df) if x not in (cols_inf+cols_y)]

cols_x_num = list(df[cols_x].select_dtypes(include=['float64', 'int64']))
# print(*cols_x_num, sep='\n')

from sklearn.feature_selection import VarianceThreshold
zero_var_filter = VarianceThreshold(threshold=0.001)
zero_var_filter.fit(df[cols_x_num])
zero_vars = list(set(df[cols_x_num].columns) \
       - set(df[cols_x_num].columns[zero_var_filter.get_support()]))
set(df[cols_x_num].columns[zero_var_filter.get_support()])
# ------------------------------------------------------------------------------
# df[zero_vars].describe()
print('\nzero_vars:', len(zero_vars), '\n',  *sorted(zero_vars), sep='\n')

df = df.drop(zero_vars, axis=1)

#%% Sort features by column names
df = df[cols_info['inf']].join(
       df.reindex(sorted(
              df.drop(cols_info['inf'], axis=1).columns
              ), axis=1
              ))

#%% INFO: Number of applications -----------------------------------------------
apps_count = df.groupby([df.first_status_day_date.dt.to_period('M'), 'customer_type'])\
    .size().unstack()\
    .assign(all=df.groupby(df.first_status_day_date.dt.to_period("M")).size())
print(apps_count)
apps_count.plot()

#%% Remove data
# with period without Credit Bureau
print(df.shape)
df = df[df['first_status_day_date']>='2018-07-01']

#%% INFO: remainings -----------------------------------------------------------
print(df.shape)
apps_count = df.groupby([df.first_status_day_date.dt.to_period('M'), 'customer_type'])\
    .size().unstack()\
    .assign(all=df.groupby(df.first_status_day_date.dt.to_period("M")).size())
print(apps_count)
apps_count.plot()

#%% [markdown] -----------------------------------------------------------------
#### Checking
#%% Check #1 -------------------------------------------------------------------
# df.loc[df.groupby(['customer_id'])['act_loan_number'].idxmax(),
#        ['customer_id', 'act_loan_number', 'y0']].head()
print(df.groupby('y0_1').size())
print(df.groupby('y0_2').size())
print(df.groupby('y0_2_1').size())
print(df.groupby('y0').size())
print(df.groupby('y0_class').size())
print(df.groupby('y1').size())
print(df.groupby('y2').size())
print(df.groupby('y3').size())
print(df.groupby('yn').size())
print(df.groupby('yd').size())

#%% Check #2 -------------------------------------------------------------------
# dcheck = \
df.loc[df.customer_id.isin([
       2597281, 2598448, 2600185, 2597298, 2597418
       # df.loc[df.act_loan_number==df.act_loan_number.max(), 'customer_id'].values[0]
       ]),
       ['customer_id', 'customer_type', 'first_status_day_date'
       ,'act_loan_number', 'act_loan_numinstal', 'act_loan_amount'
       ,'y0_1', 'y0_2', 'y0_2_1', 'y0_class', 'y0', 'y1', 'y2', 'y3', 'yd', 'yn']
].sort_values(['customer_id', 'act_loan_number'])

# dcheck.to_csv(f + 'data_out/check_df.csv',  index=False, sep=',', decimal='.', header=True)

#%% View result ----------------------------------------------------------------
df.info(verbose=True)

#%% What be splitted into train set by time
# What times periods
print(df.groupby(df.first_status_day_date.dt.to_period('M')).size()\
              .to_frame(name='period').iloc[-5:])
# How much is it in percent of whole dataset
df.groupby(df.first_status_day_date.dt.to_period('M')).size()\
       .to_frame(name='period').iloc[-5:].sum() / \
df.groupby(df.first_status_day_date.dt.to_period('M')).size()\
       .to_frame(name='period').sum() * 100

#%% Timing result --------------------------------------------------------------
print('Dataset is cleaned')
print("--- %s seconds ---" % (time.time() - start_time))

#%% Save dataset cleaned =======================================================
df.to_csv(f + 'data_in/tf/df_tf_cleaned.csv', index=False, sep=',', decimal='.',
       header=True)

#%% Result ---------------------------------------------------------------------
print('DONE. Dataset is cleaned and saved')
print("--- %s seconds ---" % (time.time() - start_time))
