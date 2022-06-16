import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption(
        "--rawdata_path",
        action="store",
        default="tests/test_rawdata.csv",
        required=False,
    )


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def random_forest_config():
    conf = {"n_estimators": 10, "max_depth": 5}
    return conf


@pytest.fixture
def rawdata_full(request):
    rawdata_path = request.config.option.rawdata_path
    df = pd.read_csv(rawdata_path)
    return df


@pytest.fixture(params=[1, 2, 42])
def rawdata_samples(request, rawdata_full):
    return rawdata_full.sample(random_state=request.param)
