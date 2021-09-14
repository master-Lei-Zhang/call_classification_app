from sklearn.base import BaseEstimator, TransformerMixin
from pandas import to_datetime

# Build a class to select features
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Sklearn transformer object to select certain columns
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.feature_names]

    def get_feature_names(self):
        return self.feature_names


# Build a class to transform "DATE_FOR" to dayofweek values
class dayofweek_transformer(BaseEstimator, TransformerMixin):
    def __init__(self): 
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        if 'DATE_FOR' in X.columns:
            X_copy['DATE_FOR']=to_datetime(X_copy['DATE_FOR']).dt.weekday                          
        self.df = X_copy
        return self.df

    def get_feature_names(self):
        return self.df.columns.tolist()