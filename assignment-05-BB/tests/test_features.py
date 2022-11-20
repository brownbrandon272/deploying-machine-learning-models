from classification_model.config.core import config
from classification_model.processing.features import FeatureTransformer


def test_feature_transformer(sample_input_data):
    # Given
    transformer = FeatureTransformer(
        model_config=config.model_config,
    )
    assert sample_input_data["Name"].iat[0] == "" ## TODO 1
    assert sample_input_data["Cabin"].iat[0] == "" ## TODO 2

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert 'Name' not in subject.data.columns
    assert 'Cabin' not in subject.data.columns
    assert 'Ticket' not in subject.data.columns

    assert subject.data["Title"].iat[0] == "" ## TODO 3
    assert subject.data["CabinLetter"].iat[0] == "" ## TODO 4
    assert subject.data["CabinNumber"].iat[0] == 0 ## TODO 5
    assert subject.data["Missing_Age"].iat[0] == 0 ## TODO 6


# def test_temporal_variable_transformer(sample_input_data):
#     # Given
#     transformer = TemporalVariableTransformer(
#         variables=config.model_config.temporal_vars,  # YearRemodAdd
#         reference_variable=config.model_config.ref_var,
#     )
#     assert sample_input_data["YearRemodAdd"].iat[0] == 1961

#     # When
#     subject = transformer.fit_transform(sample_input_data)

#     # Then
#     assert subject["YearRemodAdd"].iat[0] == 49
