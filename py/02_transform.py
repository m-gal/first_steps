# -*- coding: utf-8 -*-
"""
# Created on Dec 29 2020
    This code takes the cleaned [dwh_apps] dataset
    and makes transformation to 'credit's flows' format
@author: mikhail.galkin
"""
#%% [markdown] -----------------------------------------------------------------
#### WARNING: the code that follows would make you cry.
#### A Safety Pig is provided below for your benefit
#                            _
#    _._ _..._ .-',     _.._(`))
#   '-. `     '  /-._.-'    ',/
#      )         \            '.
#     / _    _    |             \
#    |  a    a    /              |
#    \   .-.                     ;
#     '-('' ).-'       ,'       ;
#        '-;           |      .'
#           \           \    /
#           | 7  .__  _.-\   \
#           | |  |  ``/  /`  /
#          /,_|  |   /,_/   /
#            /,_/      '`-'
#%% Define the loans' deepness of flow to back in time
n_loans = 20

#%% Load librariestiming
import time
start_time = time.time()

#%% Importing required packages
import os
import pandas as pd ;print('pandas ver.:', pd.__version__)

#%% For reflecting graphics in IPython\notebooks window correctly
# Set up needed options --------------------------------------------------------
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_info_columns', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#%% Load data
# get path
f = os.getcwd().replace(os.path.basename(os.getcwd()), '')
# load data
df = pd.read_csv(f + 'data_in/tf/df_tf_cleaned.csv', sep=',', decimal='.', header=0,
                 parse_dates=[
                     'first_status_day_date'
                 ],
                 date_parser=lambda col: pd.to_datetime(col).strftime('%Y-%m-%d')
                 , infer_datetime_format=True
                 , low_memory=False
                 , float_precision='round_trip'
                 )

#%% INFO -----------------------------------------------------------------------
# df.head()
print(df.info(), '\n')
print('#NA =', df.isna().sum().sum())
print(df.get_dtype_counts())

#%% Investigate the disribution of loan numbers --------------------------------
pd.concat([
       df.groupby(['act_loan_number']).size(),
       df.groupby(['act_loan_number']).size().cumsum()/df.shape[0]],
       axis=1, keys=['#', 'cum%']
)

#%% Sort data
df = df.sort_values(['customer_id', 'act_loan_number'])

#%% Transform the original 2D dataset to 'credits flows' format
# Define needless columns
cols_inf = ['id', 'customer_type', 'first_status_day_date' ,'act_loan_amount'
            , 'act_loan_numinstal', 'act_dpd', 'act_profit'] # w\o 'customer_id'
cols_y = ['y0_1', 'y0_2', 'y0_2_1', 'y0', 'y1', 'y2', 'y3', 'yd', 'yn'] # w\o y0_class
cols_x = [x for x in list(df) if x not in (cols_inf+cols_y+['customer_id', 'y0_class'])]

# Initiate credits flows's dataframe
dflow = df.copy()
cols_00 = dict(zip(dflow[cols_x].columns, '00_'+dflow[cols_x].columns))
dflow.rename(columns=cols_00, inplace=True)

# Create new dataframe
for i in range(-1, -n_loans, -1):
    print('Shift to', i, 'loans')
    dshift = df.drop(cols_inf+cols_y, axis=1)\
        .groupby(by=['customer_id'])\
            .shift(-i)\
                .fillna(0)
    dshift.columns = str('%02d'%-i) + '_' + dshift.columns
    dflow = pd.concat([dshift, dflow], axis=1)
print('Transormation was done!')

#%% Reorder the features
dflow = dflow.drop(['customer_id']+cols_inf, axis=1)\
    .join(dflow[cols_inf[::-1]+['customer_id']])
dflow = dflow[sorted(list(dflow), reverse=True)]

#%% INFO -----------------------------------------------------------------------
print(dflow.info(), '\n')
print('#NA =', dflow.isna().sum().sum())
print(dflow.dtypes.value_counts())

#%% Check ----------------------------------------------------------------------
dflow.loc[dflow.customer_id.isin([
                      2598448 # 13 loans
                      , 2600185
                      , 2599301 # 20 loans
                      , 2620070 # 33 loans
                      ])].T

#%% Check#2 --------------------------------------------------------------------
dflow.loc[dflow.customer_id.isin([
                      2598448 # 13 loans
                      , 2600185
                      , 2599301 # 20 loans
                      , 2620070 # 33 loans
                      ])].T
#%% Clear workspace
del(cols_inf, cols_y, cols_00, i, dshift)
#%% Timing Result --------------------------------------------------------------
print('Dataset is transformed')
print("--- %s seconds ---" % (time.time() - start_time))

#%% Save dataset ===============================================================
dflow.to_csv(f + 'data_in/tf/dflow_'+str(n_loans)+'_loans.csv', index=False, sep=','
    , decimal='.', header=True)

#%% Result ---------------------------------------------------------------------
print('DONE. Dataset is transformed and saved')
print("--- %s seconds ---" % (time.time() - start_time))
