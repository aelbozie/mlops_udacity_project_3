from pydantic import BaseModel, Field


class CensusData(BaseModel):
    age: int = Field(..., example=49)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=160187)
    education: str = Field(..., example="9th")
    education_num: int = Field(..., example=5, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-spouse-absent", alias="marital-status"
    )
    occupation: str = Field(..., example="Other-service")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=16, alias="hours-per-week")
    native_country: str = Field(..., example="Jamaica", alias="native-country")


class Income(BaseModel):
    Income: str = Field(..., example=">50K")
