# src/custom_transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------- кастомные трансформеры ---------------- #

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        # ВАЖНО: оставляем как в твоём коде для полной совместимости с pickle
        self.medians_ = X_df.median(numeric_only=False)
        self.feature_names_in_ = X_df.columns.to_list()
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).fillna(self.medians_).values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_ if input_features is None else input_features)


class ModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.modes_ = X_df.mode(dropna=True).iloc[0]
        self.feature_names_in_ = X_df.columns.to_list()
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).fillna(self.modes_).values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_ if input_features is None else input_features)


class BinaryCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = pd.DataFrame(X).columns.to_list()
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        arr = np.rint(arr)
        arr = np.clip(arr, 0, 1)
        return arr.astype(np.int8)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_ if input_features is None else input_features)


class MissingIndicatorSimple(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.feature_names_in_ = pd.DataFrame(X).columns.to_list()
        self.out_names_ = [f"{c}__was_missing" for c in self.feature_names_in_]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        return X_df.isna().astype(np.int8).values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.out_names_)
