from classification_model.config.core import config
from classification_model.processing.features import FeatureTransformer


def test_feature_transformer(sample_input_data):
    # Given
    transformer = FeatureTransformer(
        model_config=config.model_config,
    )
    assert sample_input_data["Name"].iat[0] == "Kelly, Mr. James"
    assert sample_input_data["Cabin"].iat[14] == "E31"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert "Name" not in subject.columns
    assert "Cabin" not in subject.columns
    assert "Ticket" not in subject.columns

    assert subject["Title"].iat[0] == "Mr"
    assert subject["CabinLetter"].iat[14] == "E"
    assert subject["CabinNumber"].iat[14] == 31
    assert subject["Missing_Age"].iat[0] == 0


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
