"""A simple test api using FastAPI"""
from typing import Union
import json

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

from ..ml.data import infer_from_pipeline
from ..ml.model import load_model

app = FastAPI()


def load_config():
    with open("config/api_config.json", "r") as file:
        config = json.load(file)
    return config


config = load_config()

artifacts = load_model(config["artifact_path"])


class InputModel(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        # Changes field names with hyphens as they are used in raw data
        alias_generator = lambda s: s.replace("_", "-")


@app.get("/")
async def hello_world() -> dict:
    """Hello world

    Returns
    -------
    dict

    """
    return {"data": "This is an API to interact with the Census data model"}


@app.post("/infer")
async def infer(input: InputModel) -> int:
    """Perform inference on the models

    Parameters
    ----------
    input : Union[InputModel, None], optional
        The input data

    Returns
    -------
    float
        The prediction for the input data, given the currently configured model
    """
    input_df = pd.DataFrame([input.dict(by_alias=True)])
    prediction = infer_from_pipeline(
        input_df,
        categorical_features=artifacts["categorical_features"],
        model=artifacts["model"],
        encoder=artifacts["encoder"],
        lb=artifacts["labelbinarizer"],
    )
    return int(prediction[0])
