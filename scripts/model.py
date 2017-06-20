"""This script is intended for building prediction model: it reads samples from the data file,
apply transformations to them and searches for the best parameters for prediction."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Imputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from scripts.settings import DATA_FILE, TRAIN_FILE, TEST_FILE

import warnings

warnings.filterwarnings("ignore")


class ValueImputer(BaseEstimator, TransformerMixin):
    """
    Impute missed values with particular value.
    """
    def __init__(self, value, missing_values='NaN', copy=True):
        self.value = value
        self.missing_values = missing_values
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = self._get_mask(X, self.missing_values)
        if self.copy:
            X = X.copy()
        X[mask] = self.value
        return X

    @staticmethod
    def _get_mask(X, value):
        """
        Compute the boolean mask X == missing_values.
        """
        if value == "NaN" or value is None or (isinstance(value, float) and np.isnan(value)):
            return pd.isnull(X)
        else:
            return X == value


class OnceFittedLabelBinarizer(LabelBinarizer):
    """
    Usual LabelBinarizer, but it can be fitted only once.
    """
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        super().__init__(neg_label, pos_label, sparse_output)
        self.once_fitted = False

    def fit(self, y):
        if self.once_fitted:
            return self
        self.once_fitted = True
        return super().fit(y)


class FundImputer(BaseEstimator, TransformerMixin):
    """
    Impute average funds based on total rounds using RANSACRegressor.
    """

    def __init__(self):
        self.clf = RANSACRegressor()

    def fit(self, X, y=None):
        frame = pd.DataFrame({'total_rounds': X[:, 0], 'average_funded': X[:, 1]})
        grouped = frame.groupby('total_rounds').average_funded.mean()
        rounds_funds = pd.DataFrame({'rounds': grouped.index, 'funded': grouped})
        shape = (len(rounds_funds), 1)
        self.clf.fit(rounds_funds.rounds.as_matrix().reshape(shape), rounds_funds.funded.as_matrix().reshape(shape))
        return self

    def transform(self, X):
        frame = pd.DataFrame({'total_rounds': X[:, 0], 'average_funded': X[:, 1]})
        null_funded = frame.average_funded.isnull()
        total_shape = (len(frame), 1)
        null_funded_shape = (len(frame[null_funded]), 1)
        prediction = self.clf.predict(frame[null_funded].total_rounds.as_matrix().reshape(null_funded_shape))
        frame.loc[null_funded, 'average_funded'] = prediction.ravel()
        transformed = frame.average_funded.as_matrix().reshape(total_shape)
        return transformed


class ParticipantsImputer(BaseEstimator, TransformerMixin):
    """
    Impute participants number based on average funds using RANSACRegressor.
    """

    def __init__(self):
        self.clf = RANSACRegressor()

    def fit(self, X, y=None):
        frame = pd.DataFrame({'average_funded': X[:, 0], 'average_participants': X[:, 1]})
        funds_participants = frame[(frame.average_participants != 0.0) & frame.average_funded.notnull()]
        shape = (len(funds_participants), 1)
        features = funds_participants.average_funded.as_matrix().reshape(shape)
        ground_truth = funds_participants.average_participants.as_matrix().reshape(shape)
        self.clf.fit(features, ground_truth)
        return self

    def transform(self, X):
        frame = pd.DataFrame({'average_funded': X[:, 0], 'average_participants': X[:, 1]})
        null_participants = (frame.average_participants == 0.0) & frame.average_funded.notnull()
        total_shape = (len(frame), 1)
        null_funded_shape = (len(frame[null_participants]), 1)
        prediction = self.clf.predict(frame[null_participants].average_funded.as_matrix().reshape(null_funded_shape))
        frame.loc[null_participants, 'average_participants'] = prediction.ravel()
        transformed = frame.average_participants.as_matrix().reshape(total_shape)
        return transformed


class ModelBuilder:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.clf = None
        category_binarizer = OnceFittedLabelBinarizer()
        country_binarizer = OnceFittedLabelBinarizer()
        state_binarizer = OnceFittedLabelBinarizer()
        self.category_mapper = DataFrameMapper([
            (['category_code'], [CategoricalImputer(), category_binarizer]),
            (['country_code'], [CategoricalImputer(), country_binarizer]),
            (['state_code'], [CategoricalImputer(), state_binarizer]),
        ])
        self.mapper = DataFrameMapper([
            (['category_code'], [CategoricalImputer(), category_binarizer], {'alias': 'category'}),
            (['country_code'], [CategoricalImputer(), country_binarizer], {'alias': 'country'}),
            (['state_code'], [CategoricalImputer(), state_binarizer], {'alias': 'state'}),
            (['mba_degree'], [ValueImputer(0), StandardScaler()]),
            (['phd_degree'], [ValueImputer(0), StandardScaler()]),
            (['ms_degree'], [ValueImputer(0), StandardScaler()]),
            (['other_degree'], [ValueImputer(0)]),
            (['age'], [Imputer(), StandardScaler()]),
            (['offices'], [ValueImputer(1.0), StandardScaler()]),
            (['products_number'], [ValueImputer(1.0), StandardScaler()]),
            (['average_funded', 'average_participants'], [ParticipantsImputer(), StandardScaler()],
             {'alias': 'average_participants'}),
            (['total_rounds'], None),
            (['ipo'], None),
            (['is_closed'], None),
            (['total_rounds', 'average_funded'], [FundImputer(), StandardScaler()], {'alias': 'average_funded'}),
            (['acquired_companies'], [ValueImputer(0)]),
        ])
        SVC_C_grid = [10 ** i for i in range(-3, 4)]
        SVC_gamma_grid = [10 ** i for i in range(-3, 1)] + ['auto']
        MLP_hidden_layer_sizes = [[25], [50], [75], [100], [50, 25], [75, 50], [100, 75], [75, 50, 25], [100, 75, 50]]
        MLP_activation = ['logistic', 'tanh', 'relu']
        self.grid = [{'clf': [GradientBoostingClassifier()], 'clf__n_estimators': [20 * i for i in range(5, 8)],
                 'clf__max_depth': [i + 3 for i in range(2, 6)]},
                {'clf': [SVC(kernel='rbf', class_weight='balanced')], 'clf__C': SVC_C_grid,
                 'clf__gamma': SVC_gamma_grid},
                {'clf': [SVC(kernel='poly', class_weight='balanced')], 'clf__C': SVC_C_grid,
                 'clf__gamma': SVC_gamma_grid,
                 'clf__degree': list(range(3, 6))},
                {'clf': [MLPClassifier()], 'clf__hidden_layer_sizes': MLP_hidden_layer_sizes,
                 'clf__activation': MLP_activation,
                 'clf__alpha': [10 ** i for i in range(-1, 3)]}]

    def read_data(self, data_path, train_path, test_path):
        """
        Read data.
        :param data_path: path to full data
        :param train_path: path to train data
        :param test_path: path to test data
        :return:
        """
        self.data = pd.read_csv(data_path)
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        self.X_train = train_data.drop(['company_id', 'is_acquired'], axis=1)
        self.Y_train = train_data.is_acquired.as_matrix()
        self.X_test = test_data.drop(['company_id', 'is_acquired'], axis=1)
        self.Y_test = test_data.is_acquired.as_matrix()

    def fit(self):
        """
        Find the best parameter for classification.
        :return:
        """
        self.category_mapper.fit(self.data)
        estimators = [('fill_nan', self.mapper), ('clf', GradientBoostingClassifier())]
        pipe = Pipeline(estimators)
        self.clf = GridSearchCV(pipe, self.grid, scoring='f1', cv=StratifiedKFold(n_splits=3, shuffle=True), verbose=5)
        self.clf.fit(self.X_train, self.Y_train)

    def print_results(self):
        """
        Print best score, best params and metrics for prediction for test data.
        :return:
        """
        print("Best score: ", self.clf.best_score_)
        print("Best params: ", self.clf.best_params_)
        prediction = self.clf.predict(self.X_test)
        print("F1-score for test data: ", f1_score(self.Y_test, prediction))
        print("Recall for test data: ", recall_score(self.Y_test, prediction))
        print("Precision for test data: ", precision_score(self.Y_test, prediction))


def do_model_building():
    model_builder = ModelBuilder()
    model_builder.read_data(DATA_FILE, TRAIN_FILE, TEST_FILE)
    model_builder.fit()
    model_builder.print_results()

if __name__ == '__main__':
    do_model_building()
