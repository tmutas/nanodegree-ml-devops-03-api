import pytest
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.ml.model import train_model, save_model, load_model
from starter.ml.data import process_data, infer_from_pipeline


@pytest.fixture
def train_data(rawdata_samples, cat_features, label_column):
    X, y, encoder, lb = process_data(
        rawdata_samples,
        categorical_features=cat_features,
        label=label_column,
        training=True,
    )
    return X, y, encoder, lb


@pytest.fixture
def dummy_artifacts():
    arts = {
        "model": RandomForestClassifier(),
        "binarizer": LabelBinarizer(),
        "encoder": OneHotEncoder(),
    }

    return arts


def test_train_model(train_data, random_forest_config):
    model = train_model(train_data[0], train_data[1], random_forest_config)

    # Test that desired model is returned
    assert isinstance(model, RandomForestClassifier)

    # Test that parameters are passed correctly
    for key, value in random_forest_config.items():
        assert getattr(model, key) == value


def test_save_model(tmpdir, dummy_artifacts):
    save_model(tmpdir, dummy_artifacts)

    for key, value in dummy_artifacts.items():
        filepath = Path(tmpdir) / f"{key}.pkl"

        assert filepath.exists()

        with filepath.open("rb") as fl:
            obj = pickle.load(fl)
            assert isinstance(obj, type(value))


def test_load_model(tmpdir, dummy_artifacts):
    save_model(tmpdir, dummy_artifacts)

    loaded_artifacts = load_model(tmpdir)

    # Compare that loaded_dummy_artifacts
    for key, value in dummy_artifacts.items():
        assert isinstance(loaded_artifacts[key], type(value))


def test_inference(
    rawdata_samples, train_data, random_forest_config, cat_features, label_column
):
    X_train, y_train, encoder, lb = train_data
    model = train_model(X_train, y_train, random_forest_config)

    X_raw = rawdata_samples.drop(columns=label_column).head(1)

    inference = infer_from_pipeline(
        X_raw,
        categorical_features=cat_features,
        model=model,
        encoder=encoder,
        lb=lb,
    )

    assert isinstance(inference, np.ndarray)
    # Testing a classfier, has to return 0 or 1
    assert inference[0] in (0, 1)
