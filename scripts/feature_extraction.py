"""This script forms csv data file with characteristics described in "Data description.md"
from sql dump provided by CrunchBase and located in "sql" folder.
The dump should be imported in local MySQL database.
In addition, this script splits data into two parts: train_data and test_data.
You will be asked for user, password and scheme for database you use."""

import pandas as pd
import MySQLdb as sql
import numpy as np
from scripts.sql_requests import get_degrees_query, get_financial_ipo_offices_products_query, valid_degrees
from sklearn.model_selection import train_test_split
from .settings import DATA_FILE, TEST_FILE, TRAIN_FILE


def fix_column(df, column):
    """
    Set "column" in df to NaN if "column" was after acquisition.
    """
    column_after_acquired = '%s_after_acquired' % column
    column_acquired = pd.DataFrame({column: df[column], 'acquired': df.acquired_at})
    column_acquired.dropna(how='any', inplace=True)
    column_acquired.loc[:, column] = column_acquired[column].apply(pd.to_datetime)
    column_acquired.loc[:, 'acquired'] = column_acquired.acquired.apply(pd.to_datetime)
    column_acquired[column_after_acquired] = (column_acquired[column] - column_acquired['acquired']).apply(
        lambda x: x.days > 0)
    indexes = column_acquired[column_acquired[column_after_acquired]].index
    df.loc[indexes, column] = np.nan


def calculate_age(df):
    """
    Add an "age" column to df where age is set up on date of acquisition, if it was,
    or on 01.01.2014.
    """
    is_acquired = df.is_acquired
    is_not_acquired = ~is_acquired
    df.loc[is_acquired, 'age'] = (
        df[is_acquired]['acquired_at'].apply(pd.to_datetime) - df[is_acquired]['founded_at'].apply(pd.to_datetime)) \
        .apply(lambda x: x.days)
    df.loc[is_not_acquired, 'age'] = (
        pd.to_datetime('2014-01-01') - df[is_not_acquired]['founded_at'].apply(pd.to_datetime)) \
        .apply(lambda x: x.days)
    df.drop(df[df.age < 0].index, inplace=True)


if __name__ == '__main__':
    user = input("user: ")
    password = input("password: ")
    scheme = input("scheme: ")

    db = sql.connect(user=user, passwd=password, db=scheme)
    data = get_financial_ipo_offices_products_query(db, scheme)
    degrees = get_degrees_query(db, scheme)

    for degree_name in valid_degrees + ('other',):
        deg = degrees[degrees.degree_type == degree_name][['count', 'company_id']]
        deg.columns = ['%s_degree' % degree_name, 'company_id']
        data = data.merge(deg, on='company_id', how='left')

    fix_column(data, 'public_at')
    fix_column(data, 'closed_at')

    data['ipo'] = data.public_at.notnull()
    data['is_acquired'] = data.acquired_at.notnull()
    data['is_closed'] = data.closed_at.notnull()

    calculate_age(data)

    data.loc[:, 'country_code'] = data['country_code'].apply(lambda x: x if x in ['USA', 'NZL'] else 'other')
    data.loc[:, 'state_code'] = data['state_code'].apply(lambda x: 'California' if x == 'CA' else 'other')

    columns_to_drop = ['founded_at', 'closed_at', 'public_at', 'acquired_at', 'city', 'region']
    data.drop(columns_to_drop, inplace=True, axis=1)
    data.to_csv(DATA_FILE, index=False)

    x_train, x_test, y_train, y_test = train_test_split(data.drop(['is_acquired'], axis=1), data['is_acquired'],
                                                        stratify=data['is_acquired'], test_size=0.2)
    x_train['is_acquired'] = y_train
    x_test['is_acquired'] = y_test

    x_train.to_csv(TRAIN_FILE, index=False)
    x_test.to_csv(TEST_FILE, index=False)
