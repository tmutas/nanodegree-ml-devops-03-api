import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


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
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)

    return precision, recall, fbeta


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
