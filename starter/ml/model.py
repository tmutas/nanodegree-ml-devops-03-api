import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, params):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    params : dict
        Dictionary
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf = RandomForestClassifier(
        **params,
    )

    rf = rf.fit(X_train, y_train)

    return rf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    dict
        Metrics values for "precision", "recall" and "fbeta" as keys
        and "n" denoting sample size
    """
    metrics = {}
    metrics["precision"] = precision_score(y, preds, zero_division=1)
    metrics["recall"] = recall_score(y, preds, zero_division=1)
    metrics["fbeta"] = fbeta_score(y, preds, beta=1, zero_division=1)
    metrics["n"] = len(y)
    return metrics


def compute_slice_metrics(df_raw: pd.DataFrame, col: str, artifacts: dict, label: str):
    """Computes metrics on slices based on a categorical column

    Parameters
    ----------
    df_raw : pd.DataFrame
        The raw dataframce with feature columns and known labels
    col : str
        Column name on which to slice the data on
    artifacts : dict
        Containing model, encoder, labelbinarizer and categorical_features
        Necesseary for the processing pipeline
    label : str
        Label column name

    Returns
    -------
    dict
        For each unique value of col, contains a dict with metric keys and values
    """
    col_values = df_raw[col].unique()
    col_metrics = {}
    for val in col_values:
        X, y, *_ = process_data(
            df_raw[df_raw.loc[:, col] == val],
            categorical_features=artifacts["categorical_features"],
            label=label,
            training=False,
            encoder=artifacts["encoder"],
            lb=artifacts["labelbinarizer"],
        )
        pred = artifacts["model"].predict(X)
        col_metrics[val] = compute_model_metrics(y, pred)

    return col_metrics


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    return model.predict(X)


def save_model(path, artifacts: dict):
    """Saves artifacts to given directory as pickles

    Parameters
    ----------
    path : path-like
        Path or string to directory, if does not exists will create
    artifacts : dict
        Dict of name:artifact pairs. Names will be used as filenames
    """
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True)

    for name, art in artifacts.items():
        art_path = path / f"{name}.pkl"
        with art_path.open("wb+") as fl:
            pickle.dump(art, fl)


def load_model(
    path,
) -> dict:
    """Load model artifacts from a directory with .pkl files

    Parameters
    ----------
    path : path-like
        Path to a directory, containing pickled artifacts

    Returns
    -------
    dict
        dict with name:artifact pairs

    Raises
    ------
    NotADirectoryError
        If path is not a directory
    """
    path = Path(path)

    if not path.is_dir():
        raise NotADirectoryError()

    artifacts = {}

    for file in path.glob("*.pkl"):
        name = file.name.replace(".pkl", "")
        with file.open("rb") as fl:
            artifacts[name] = pickle.load(fl)

    return artifacts
