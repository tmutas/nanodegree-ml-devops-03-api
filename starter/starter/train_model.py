"""Script to train machine learning model"""
from argparse import ArgumentParser
from pathlib import Path
import json
import logging
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, save_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# Add code to load in the data.
def run(args):
    if args.model_params is not None:
        train_params = json.load(args.model_params)
    else:
        train_params = {}

    rawdata = pd.read_csv(args.rawdata)

    logging.debug("Loaded rawdata dataframe")

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(rawdata, test_size=0.20)

    logging.debug("Train test split performed")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.

    X_test, y_test, encoder_test, lb_test = process_data(
        test, categorical_features=cat_features, label="salary", training=True
    )

    logging.debug("Train and test data processed")

    # Train and save a model.

    train_params = {"n_estimators": 5, "max_depth": 2}
    model = train_model(X_train, y_train, train_params)

    logging.debug("Model trained")

    artifacts = {"model": model, "encoder": encoder, "labelbinarizer": lb}
    if args.artifact_path is not None:
        save_model(args.artifact_path, artifacts)
        logging.debug(f"Artifacts saved to {args.artifact_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "rawdata",
        type=Path,
        help="Path to the rawdata csv file",
    )

    parser.add_argument(
        "--artifact_path",
        type=Path,
        help="Directory to where model artifacts are to be saved, does not save by default",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--model_params",
        type=Path,
        help=(
            "File path to a json that contains model parameters"
            "that can be passed to sklearn's RandomForestClassifier"
        ),
        default=None,
    )

    args = parser.parse_args()

    run(args)
