 src/custom_imputers.py
from typing import List, Sequence, Optional, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.medians_ = X_df.median(numeric_only=False)
        self.feature_names_in_ = X_df.columns.to_list()
        return self
    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).fillna(self.medians_).values
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_ if input_features is None else input_features)
