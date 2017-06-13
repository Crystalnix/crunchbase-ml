import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RANSACRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Imputer, FunctionTransformer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings

warnings.filterwarnings("ignore")


class ValueImputer(BaseEstimator, TransformerMixin):
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
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        super().__init__(neg_label, pos_label, sparse_output)
        self.once_fitted = False

    def fit(self, y):
        if self.once_fitted:
            return self
        self.once_fitted = True
        return super().fit(y)

    def transform(self, y):
        transformed = super().transform(y)
        return transformed


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


data = pd.read_csv('../data/data.csv')
category_binarizer = OnceFittedLabelBinarizer()
country_binarizer = OnceFittedLabelBinarizer()
state_binarizer = OnceFittedLabelBinarizer()
category_mapper = DataFrameMapper([
    (['category_code'], [CategoricalImputer(), category_binarizer]),
    (['country_code'], [CategoricalImputer(), country_binarizer]),
    (['state_code'], [CategoricalImputer(), state_binarizer]),
])
category_mapper.fit(data)

log_transformer = FunctionTransformer(np.log)

mapper = DataFrameMapper([
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
])
SVC_C_grid = [10 ** i for i in range(-3, 4)]
SVC_gamma_grid = [10 ** i for i in range(-3, 1)] + ['auto']
MLP_hidden_layer_sizes = [[25], [50], [75], [100], [50, 25], [75, 50], [100, 75], [75, 50, 25], [100, 75, 50]]
MLP_activation = ['logistic', 'tanh', 'relu']
grid = [{'clf': [GradientBoostingClassifier()], 'clf__n_estimators': [20 * i for i in range(5, 8)],
         'clf__max_depth': [i + 3 for i in range(2, 6)]},
        {'clf': [SVC(kernel='rbf', class_weight='balanced')], 'clf__C': SVC_C_grid, 'clf__gamma':SVC_gamma_grid},
        {'clf': [SVC(kernel='poly', class_weight='balanced')], 'clf__C': SVC_C_grid, 'clf__gamma':SVC_gamma_grid,
         'clf__degree': list(range(3, 6))},
        {'clf': [MLPClassifier()], 'clf__hidden_layer_sizes': MLP_hidden_layer_sizes, 'clf__activation': MLP_activation,
         'clf__alpha': [10 ** i for i in range(-1, 3)]}]

train_data = pd.read_csv('../data/train_data.csv')
test_data = pd.read_csv('../data/test_data.csv')
X_train = train_data.drop(['company_id', 'is_acquired'], axis=1)
Y_train = train_data.is_acquired.as_matrix()
X_test = test_data.drop(['company_id', 'is_acquired'], axis=1)
Y_test = test_data.is_acquired.as_matrix()

estimators = [('fill_nan_log_transform', mapper), ('clf', LogisticRegression(solver='sag'))]
pipe = Pipeline(estimators)
clf = GridSearchCV(pipe, grid, scoring='f1', cv=StratifiedKFold(n_splits=3, shuffle=True), verbose=5)
clf.fit(X_train, Y_train)

print("Best score: ", clf.best_score_)
print("Best params: ", clf.best_params_)

prediction = clf.predict(X_test)
print(cross_val_score(clf, X_test, Y_test, scoring='f1'))

tp = (Y_test & prediction).sum() / Y_test.sum()
fp = (~Y_test & prediction).sum() / prediction.sum()
print("tp: ", tp)
print("fp: ", fp)
