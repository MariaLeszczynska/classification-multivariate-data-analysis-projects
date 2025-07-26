import pandas as pd
import numpy as np

from typing import List, Optional
from scipy import stats
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler,
    MinMaxScaler
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
    

    
class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean", columns: Optional[List[str]] = None, imputer = SimpleImputer):
        self.strategy = strategy
        self.columns = columns if columns is not None else []
        if imputer == SimpleImputer:
            self.imputer = imputer(strategy=self.strategy)
        elif imputer == KNNImputer:
            self.imputer = imputer(n_neighbors=self.strategy)
    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])
        return X_transformed



class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        means = X[numeric_columns].mean(axis=0)
        stds = X[numeric_columns].std(axis=0)
        z_scores = np.abs((X[numeric_columns] - means) / stds)
        outlier_mask = z_scores > self.threshold
        cleaned_data = X[~np.any(outlier_mask, axis=1)]
        return cleaned_data

    

class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed
    


class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].replace(0, np.nan)
        return X


class CustomSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, strategy="mean"):
        self.columns = columns
        self.strategy = strategy
        self.imputer = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        if self.imputer is None:
            raise RuntimeError("CustomSimpleImputer is not fitted yet.")
        X = X.copy()
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X
    


class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[column])
        return X_transformed

    


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = OneHotEncoder(sparse_output=False)
            self.encoders[column].fit(X[[column]])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            encoded = pd.DataFrame(
                self.encoders[column].transform(X[[column]]),
                columns=self.encoders[column].get_feature_names_out([column]),
                index=X.index,
            )
            X_transformed = pd.concat(
                [X_transformed.drop(columns=column), encoded], axis=1
            )
        return X_transformed

    


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed

    


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.mapping = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns

        if y is not None and y.name in X.columns:
            for col in self.columns:
                self.mapping[col] = X.groupby(col)[y.name].mean()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in self.columns:
            if col in self.mapping:
                X_encoded[col] = X_encoded[col].map(self.mapping[col])
        return X_encoded
    

# Class to transform numerical column of values into their absolute values
class AbsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].abs()
        return X
    

class CustomOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.numeric_cols = None
        self._outliers = None

    # This function identifies the numerical columns
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns
        return self

    def transform(self, X):
        if self.numeric_cols is None:
            raise ValueError("Call 'fit' before 'transform'.")

        # Make a copy of numerical columns
        X_transformed = X.copy()

        z_scores = stats.zscore(X_transformed[self.numeric_cols])

        # Concat with non-numerical columns
        self._outliers = (abs(z_scores) > self.threshold).any(axis=1)
        return X_transformed[~self._outliers]

    @property
    def outliers(self):
        return self._outliers
