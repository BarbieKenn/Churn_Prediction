from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, last_date_col='Last_Visit_Date', last_out_col='Days_Since_Last_Visit', reference_date='2025-05-30', join_date_col='Join_Date', join_out_col='Membership_Days'):
        self.last_date_col = last_date_col
        self.last_out_col = last_out_col
        self.reference_date = reference_date
        self.join_date_col = join_date_col
        self.join_out_col = join_out_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.last_date_col] = pd.to_datetime(X_copy[self.last_date_col])
        ref_date = pd.Timestamp(self.reference_date)
        X_copy[self.last_out_col] = (ref_date - X_copy[self.last_date_col]).dt.days.clip(lower=0)

        X_copy[self.join_date_col] = pd.to_datetime(X_copy[self.join_date_col])
        X_copy[self.join_out_col] = (ref_date - X_copy[self.join_date_col]).dt.days.clip(lower=0)

        return X_copy.drop(columns=[f'{self.last_date_col}', f'{self.join_date_col}'])
