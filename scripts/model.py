import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RANSACRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Imputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.base import BaseEstimator, TransformerMixin


class ValueImputer(CategoricalImputer):
    def __init__(self, value, missing_values='NaN', copy=True):
        self.fill_ = value
        super().__init__(missing_values, copy)

    def fit(self, X, y=None):
        return self


class OnceFittedLabelBinarizer(LabelBinarizer):
    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        super().__init__(neg_label, pos_label, sparse_output)
        self.once_fitted = False

    def fit(self, y):
        if self.once_fitted:
            return self
        self.once_fitted = True
        return super().fit(y)


class FundInputer(BaseEstimator, TransformerMixin):
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
        return frame.average_funded.as_matrix().reshape(total_shape)


data = pd.read_csv('../data/data.csv')
category_binarizer = OnceFittedLabelBinarizer()
category_mapper = DataFrameMapper([
   (['category_code'], [CategoricalImputer(), category_binarizer])
])
category_mapper.fit(data)

mapper = DataFrameMapper([
    (['category_code'], [CategoricalImputer(), category_binarizer], {'alias': 'category'}),
    (['country_code'], LabelBinarizer(), {'alias': 'country'}),
    (['state_code'], LabelBinarizer(), {'alias': 'state'}),
    (['mba_degree'], ValueImputer(0)),
    (['phd_degree'], ValueImputer(0)),
    (['ms_degree'], ValueImputer(0)),
    (['other_degree'], ValueImputer(0)),
    (['age'], Imputer()),
    (['offices'], ValueImputer(1.0)),
    (['products_number'], ValueImputer(1.0)),
    (['average_participants'], None),
    (['total_rounds'], None),
    (['ipo'], None),
    (['is_closed'], None),
    (['total_rounds', 'average_funded'], FundInputer(), {'alias': 'average_funded'}),
])

train_data = pd.read_csv('../data/train_data.csv')
transformed = mapper.fit_transform(train_data)
print(np.isnan(transformed).any())
print(mapper.transformed_names_)
