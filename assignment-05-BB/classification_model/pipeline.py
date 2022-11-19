# from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
# from feature_engine.imputation import (
#     AddMissingIndicator,
#     CategoricalImputer,
#     MeanMedianImputer,
# )
# from feature_engine.selection import DropFeatures
# from feature_engine.transformation import LogTransformer
# from feature_engine.wrappers import SklearnTransformerWrapper
# from sklearn.linear_model import Lasso
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import Binarizer, MinMaxScaler

from classification_model.config.core import config
from classification_model.processing import features as pp

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline # , FeatureUnion
from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import cross_val_score
import lightgbm as lgb

def create_pipeline():
    # Define categorical pipeline
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        # ignored unkowns would be given a row of all zeroes, but labelEncoding() in FeatureTransformer() is already assigning them class "0"
    ])

    # Define numerical pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # config = Config()
    # Combine categorical and numerical pipelines
    columnTransform = ColumnTransformer([
        ('cat', cat_pipe, config.model_config.categorical_features_select),
        ('num', num_pipe, config.model_config.numerical_features_select)
    ])

    preprocessor = Pipeline([
        ('features', pp.FeatureTransformer(model_config=config.model_config)),
        ('columnTransform', columnTransform)
    ])

    # Fit a pipeline with transformers and an estimator to the training data
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMClassifier(learning_rate=0.025))
    ])
    return pipe

price_pipe = create_pipeline()

# price_pipe = Pipeline(
#     [
#         # ===== IMPUTATION =====
#         # impute categorical variables with string missing
#         (
#             "missing_imputation",
#             CategoricalImputer(
#                 imputation_method="missing",
#                 variables=config.model_config.categorical_vars_with_na_missing,
#             ),
#         ),
#         (
#             "frequent_imputation",
#             CategoricalImputer(
#                 imputation_method="frequent",
#                 variables=config.model_config.categorical_vars_with_na_frequent,
#             ),
#         ),
#         # add missing indicator
#         (
#             "missing_indicator",
#             AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
#         ),
#         # impute numerical variables with the mean
#         (
#             "mean_imputation",
#             MeanMedianImputer(
#                 imputation_method="mean",
#                 variables=config.model_config.numerical_vars_with_na,
#             ),
#         ),
#         # == TEMPORAL VARIABLES ====
#         (
#             "elapsed_time",
#             pp.TemporalVariableTransformer(
#                 variables=config.model_config.temporal_vars,
#                 reference_variable=config.model_config.ref_var,
#             ),
#         ),
#         ("drop_features", DropFeatures(features_to_drop=[config.model_config.ref_var])),
#         # ==== VARIABLE TRANSFORMATION =====
#         ("log", LogTransformer(variables=config.model_config.numericals_log_vars)),
#         (
#             "binarizer",
#             SklearnTransformerWrapper(
#                 transformer=Binarizer(threshold=0),
#                 variables=config.model_config.binarize_vars,
#             ),
#         ),
#         # === mappers ===
#         (
#             "mapper_qual",
#             pp.Mapper(
#                 variables=config.model_config.qual_vars,
#                 mappings=config.model_config.qual_mappings,
#             ),
#         ),
#         (
#             "mapper_exposure",
#             pp.Mapper(
#                 variables=config.model_config.exposure_vars,
#                 mappings=config.model_config.exposure_mappings,
#             ),
#         ),
#         (
#             "mapper_finish",
#             pp.Mapper(
#                 variables=config.model_config.finish_vars,
#                 mappings=config.model_config.finish_mappings,
#             ),
#         ),
#         (
#             "mapper_garage",
#             pp.Mapper(
#                 variables=config.model_config.garage_vars,
#                 mappings=config.model_config.garage_mappings,
#             ),
#         ),
#         # == CATEGORICAL ENCODING
#         (
#             "rare_label_encoder",
#             RareLabelEncoder(
#                 tol=0.01, n_categories=1, variables=config.model_config.categorical_vars
#             ),
#         ),
#         # encode categorical variables using the target mean
#         (
#             "categorical_encoder",
#             OrdinalEncoder(
#                 encoding_method="ordered",
#                 variables=config.model_config.categorical_vars,
#             ),
#         ),
#         ("scaler", MinMaxScaler()),
#         (
#             "Lasso",
#             Lasso(
#                 alpha=config.model_config.alpha,
#                 random_state=config.model_config.random_state,
#             ),
#         ),
#     ]
# )
