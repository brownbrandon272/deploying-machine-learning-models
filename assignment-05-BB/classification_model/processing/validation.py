from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.original_features
        if var
        not in config.model_config.numerical_vars_with_na
        + config.model_config.categorical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    relevant_data = input_data[config.model_config.original_features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        TitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class PassengerDataInputSchema(BaseModel):
    Age: Optional[float]
    Cabin: Optional[str]
    Embarked: Optional[str]
    Fare: Optional[float]
    Name: Optional[str]
    Parch: Optional[int]
    Pclass: Optional[int]
    Sex: Optional[str]
    SibSp: Optional[int]
    Ticket: Optional[str]


class TitanicDataInputs(BaseModel):
    inputs: List[PassengerDataInputSchema]
