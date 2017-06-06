
"""This script forms csv data file with characteristics described in "Data description.md"
from sql dump provided by CrunchBase and located in "sql" folder.
The dump should be imported in local MySQL database.
In addition, this script splits data into two parts: train_data and test_data.
You will be asked for user, password and scheme for database you use."""

import pandas as pd
import MySQLdb as sql
import numpy as np
from scripts.sql_requests import financial_ipo_offices_products_request, degrees_request
from sklearn.model_selection import train_test_split


def fix_ipo(df):
    """Set IPO information to NaN if IPO was after acquisition."""
    ipo_fields = ['public_at']
    ipo = pd.DataFrame({'ipo': df.public_at, 'acquired': df.acquired_at})
    ipo.dropna(how='any', inplace=True)
    ipo.loc[:, 'ipo'] = ipo.ipo.apply(pd.to_datetime)
    ipo.loc[:, 'acquired'] = ipo.acquired.apply(pd.to_datetime)
    ipo['ipo_after_acquired'] = (ipo['ipo'] - ipo['acquired']).apply(lambda x: x.days > 0)
    indexes = ipo[ipo['ipo_after_acquired']].index
    df.loc[indexes, ipo_fields] = np.nan


def fix_closed_at(df):
    """Set closed_at to NaN if closed after acquisition."""
    closed = pd.DataFrame({'closed': df.closed_at, 'acquired': df.acquired_at})
    closed.dropna(how='any', inplace=True)
    closed.loc[:, 'closed'] = closed.closed.apply(pd.to_datetime)
    closed.loc[:, 'acquired'] = closed.acquired.apply(pd.to_datetime)
    closed['closed_after_acquired'] = (closed['closed'] - closed['acquired']).apply(lambda x: x.days > 0)
    indexes = closed[closed['closed_after_acquired']].index
    df.loc[indexes, 'closed_at'] = np.nan

user = input("user: ")
password = input("password: ")
scheme = input("scheme: ")

db = sql.connect(user=user, passwd=password, db=scheme)
df = pd.read_sql(financial_ipo_offices_products_request.format(scheme), con=db)
valid_degrees = ('mba', 'phd', 'ms')

degrees = pd.read_sql(degrees_request.format(scheme, valid_degrees), con=db)

for degree_name in valid_degrees + ('other',):
    deg = degrees[degrees.degree_type == degree_name][['count', 'company_id']]
    deg.columns = ['%s_degree' % degree_name, 'company_id']
    df = df.merge(deg, on='company_id', how='left')

fix_ipo(df)
fix_closed_at(df)

df['ipo'] = df.public_at.notnull()
df['is_acquired'] = df.acquired_at.notnull()
df['is_closed'] = df.closed_at.notnull()

is_acquired = df.is_acquired == True
is_not_acquired = ~is_acquired
df.loc[is_acquired, 'age'] = (
    df[is_acquired]['acquired_at'].apply(pd.to_datetime) - df[is_acquired]['founded_at'].apply(pd.to_datetime)) \
    .apply(lambda x: x.days)
df.loc[is_not_acquired, 'age'] = (
    pd.to_datetime('2014-01-01') - df[is_not_acquired]['founded_at'].apply(pd.to_datetime)) \
    .apply(lambda x: x.days)

columns_to_drop = ['founded_at', 'closed_at', 'public_at', 'acquired_at', 'city', 'region']
df.loc[:, 'country_code'] = df['country_code'].apply(lambda x: x if x in ['USA', 'NZL'] else 'other')
df.loc[:, 'state_code'] = df['state_code'].apply(lambda x: 'California' if x == 'CA' else 'other')

df.drop(columns_to_drop, inplace=True, axis=1)
df.to_csv('../data/data.csv', index=False)

x_train, x_test, y_train, y_test = train_test_split(df.drop(['is_acquired'], axis=1), df['is_acquired'],
                                                    stratify=df['is_acquired'], test_size=0.2)
x_train['is_acquired'] = y_train
x_test['is_acquired'] = y_test

x_train.to_csv('../data/train_data.csv', index=False)
x_test.to_csv('../data/test_data.csv', index=False)

