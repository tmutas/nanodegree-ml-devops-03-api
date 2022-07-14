import starter.ml.data as data

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def test_train_test_split(rawdata_samples, cat_features):
    X, y, encoder, lb = data.process_data(
        rawdata_samples,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert len(X) == len(y)

    # Test dimensions of processed data
    assert len(X.shape) == 2
    assert len(y.shape) == 1

    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
