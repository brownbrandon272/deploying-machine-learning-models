from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
            # missing_prefix:str,
            # title_var:str, title_create_var_str:str,
            # cabin_var:str, cabin_create_var_str:str, cabin_create_var_int:int,
            model_config:object,
            verbose = False,
            ):
        """
        Constructor Function.
        Because I will be using this in a pipeline, __init__() is
        called when pipeline is first created.
        """
        self.config = model_config
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        self.data = X.copy()
        self.createMissing()
        self.createTitle()
        self.createCabin()
        return self.data

    def selectFeatures(self):
        cols = self.config.cat_vars + self.config.cont_vars
        self.data = self.data[cols]
        return self
        
    def createMissing(self):
        for x in self.config.numerical_vars_with_na:
            self.data[self.config.missing_prefix + x] = self.data[x].isna() + 0
        return self

    def createTitle(self):
        ## Name -> Title
        self.data[self.config.title_var] = \
            self.data[self.config.title_create_var_str].apply(lambda x: x[x.find(',')+2 : x.find('.')])
        return self

    def createCabin(self):
        def _get_cabin_number(cabin):
            try:
                return int(cabin.split()[-1][1:])
            except:
                return None
        ## Cabin -> Cabin Letter and Cabin Number
        self.data[self.config.cabin_create_var_str] = self.data[self.config.cabin_var].str.slice(0,1)
        self.data[self.config.cabin_create_var_int] = self.data[self.config.cabin_var].apply(_get_cabin_number)
        return self
        
    def labelEncoding(self):
        for col, enc_dict in zip(self.config.ENC_DICTS.keys(), self.config.ENC_DICTS.values()):
            self.data[col] = self.data[col] \
                .map(enc_dict) \
                .fillna(0) \
                .replace({-1: np.nan})
        return self


# class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
#     """Temporal elapsed time transformer."""

#     def __init__(self, variables: List[str], reference_variable: str):

#         if not isinstance(variables, list):
#             raise ValueError("variables should be a list")

#         self.variables = variables
#         self.reference_variable = reference_variable

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need this step to fit the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:

#         # so that we do not over-write the original dataframe
#         X = X.copy()

#         for feature in self.variables:
#             X[feature] = X[self.reference_variable] - X[feature]

#         return X


# class Mapper(BaseEstimator, TransformerMixin):
#     """Categorical variable mapper."""

#     def __init__(self, variables: List[str], mappings: dict):

#         if not isinstance(variables, list):
#             raise ValueError("variables should be a list")

#         self.variables = variables
#         self.mappings = mappings

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need the fit statement to accomodate the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].map(self.mappings)

#         return X
