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
from scripts.settings import DATA_FILE, TEST_FILE, TRAIN_FILE


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


class FeatureExtractor:
    def __init__(self):
        self.db = None
        self.data = None
        self.degrees = None

    def connect_to_db(self, user, password, scheme):
        """
        Connect to database.
        :param user: database user
        :param password: database user's password
        :param scheme: database scheme in which sql dump is stored
        :return:
        """
        self.db = sql.connect(user=user, passwd=password, db=scheme)
        self.scheme = scheme

    def perform_queries(self):
        """
        Extract data from database.
        :return:
        """
        self.data = get_financial_ipo_offices_products_query(self.db, self.scheme)
        self.degrees = get_degrees_query(self.db, self.scheme)

    def extract_degrees(self):
        """
        Calculate how much people with degrees stored in valid_degrees are associated with each company.
        :return:
        """
        degrees = self.degrees
        for degree_name in valid_degrees + ('other',):
            deg = degrees[degrees.degree_type == degree_name][['count', 'company_id']]
            deg.columns = ['%s_degree' % degree_name, 'company_id']
            self.data = self.data.merge(deg, on='company_id', how='left')

    def fix_public_and_closed(self):
        """
        Set 'public_at, 'closed_at' to NaN if they were after acquisition.
        :return:
        """
        fix_column(self.data, 'public_at')
        fix_column(self.data, 'closed_at')

    def set_binary_columns(self):
        """
        Set 'ipo', 'is_acquired', 'is_closed' in data.
        :return:
        """
        self.data['ipo'] = self.data.public_at.notnull()
        self.data['is_acquired'] = self.data.acquired_at.notnull()
        self.data['is_closed'] = self.data.closed_at.notnull()

    def calculate_age(self):
        """
        Add an "age" column to data where age is set up on date of acquisition, if it was,
        or on 01.01.2014.
        """
        df = self.data
        is_acquired = df.is_acquired
        is_not_acquired = ~is_acquired
        df.loc[is_acquired, 'age'] = (
            df[is_acquired]['acquired_at'].apply(pd.to_datetime) - df[is_acquired]['founded_at'].apply(pd.to_datetime))\
            .apply(lambda x: x.days)
        df.loc[is_not_acquired, 'age'] = (
            pd.to_datetime('2014-01-01') - df[is_not_acquired]['founded_at'].apply(pd.to_datetime)) \
            .apply(lambda x: x.days)
        df.drop(df[df.age < 0].index, inplace=True)

    def set_geo_info(self):
        """
        Set all values, except 'USA' and 'NZL' in country_code to 'other'.
        Also set state_code to 'California', if it was 'CA', else 'other'.
        :return:
        """
        data = self.data
        data.loc[:, 'country_code'] = data['country_code'].apply(lambda x: x if x in ['USA', 'NZL'] else 'other')
        data.loc[:, 'state_code'] = data['state_code'].apply(lambda x: 'California' if x == 'CA' else 'other')

    def drop_excess_columns(self):
        """
        Drop 'founded_at', 'closed_at', 'public_at', 'acquired_at', 'city', 'region' from data.
        :return:
        """
        columns_to_drop = ['founded_at', 'closed_at', 'public_at', 'acquired_at', 'city', 'region']
        self.data.drop(columns_to_drop, inplace=True, axis=1)

    def save_to_files(self):
        """
        Save resultant data to DATA_FILE, also save it to TRAIN_FILE and  TEST_FILE using stratifying strategy.
        :return:
        """
        data = self.data
        data.to_csv(DATA_FILE, index=False)
        x_train, x_test, y_train, y_test = train_test_split(data.drop(['is_acquired'], axis=1), data['is_acquired'],
                                                            stratify=data['is_acquired'], test_size=0.2)
        x_train['is_acquired'] = y_train
        x_test['is_acquired'] = y_test
        x_train.to_csv(TRAIN_FILE, index=False)
        x_test.to_csv(TEST_FILE, index=False)


def do_feature_extraction(user, password, scheme):
    """
    Usual workflow for this script.
    :param user: database user
    :param password: database user's password
    :param scheme: database scheme in which sql dump is stored
    :return:
    """
    try:
        feature_extractor = FeatureExtractor()
        feature_extractor.connect_to_db(user, password, scheme)
        feature_extractor.perform_queries()
        feature_extractor.extract_degrees()
        feature_extractor.fix_public_and_closed()
        feature_extractor.set_binary_columns()
        feature_extractor.calculate_age()
        feature_extractor.set_geo_info()
        feature_extractor.drop_excess_columns()
        feature_extractor.save_to_files()
    except sql.OperationalError:
        print("Wrong credentials for access to database.")

if __name__ == '__main__':
    user = input("user: ")
    password = input("password: ")
    scheme = input("scheme: ")
    do_feature_extraction(user, password, scheme)