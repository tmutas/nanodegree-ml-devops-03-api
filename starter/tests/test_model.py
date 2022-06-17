import pytest
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.ml.model import train_model, save_model, load_model
from starter.ml.data import process_data


@pytest.fixture
def train_data(rawdata_samples, cat_features):
    *train_data, encoder, lb = process_data(
        rawdata_samples,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return train_data


@pytest.fixture
def artifacts():
    arts = {
        "model": RandomForestClassifier(),
        "binarizer": LabelBinarizer(),
        "encoder": OneHotEncoder(),
    }

    return arts


def test_train_model(train_data, random_forest_config):
    model = train_model(*train_data, random_forest_config)

    # Test that desired model is returned
    assert isinstance(model, RandomForestClassifier)

    # Test that parameters are passed correctly
    for key, value in random_forest_config.items():
        assert getattr(model, key) == value


def test_save_model(tmpdir, artifacts):
    save_model(tmpdir, artifacts)

    for key, value in artifacts.items():
        filepath = Path(tmpdir) / f"{key}.pkl"

        assert filepath.exists()

        with filepath.open("rb") as fl:
            obj = pickle.load(fl)
            assert isinstance(obj, type(value))


def test_load_model(tmpdir, artifacts):
    save_model(tmpdir, artifacts)

    loaded_artifacts = load_model(tmpdir, list(artifacts.keys()))

    # Compare that loaded_artifacts
    for key, value in artifacts.items():
        assert isinstance(loaded_artifacts[key], type(value))
