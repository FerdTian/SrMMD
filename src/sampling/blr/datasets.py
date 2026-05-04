import gzip
import io
import os
import urllib.request

import numpy as np

BLR_DATASETS = ("breast_cancer", "ionosphere", "german_credit", "covtype")


def _infer_numeric_and_categorical_columns(X):
    X = np.asarray(X)
    numeric_cols, categorical_cols = [], []
    for idx in range(X.shape[1]):
        column = X[:, idx]
        try:
            column.astype(np.float64)
            numeric_cols.append(idx)
        except (TypeError, ValueError):
            categorical_cols.append(idx)
    return numeric_cols, categorical_cols


def _one_hot_encode_train_test(X_train_cat, X_test_cat):
    train_blocks = []
    test_blocks = []
    for col_idx in range(X_train_cat.shape[1]):
        categories = list(dict.fromkeys(X_train_cat[:, col_idx].astype(str)))
        mapping = {category: idx for idx, category in enumerate(categories)}

        train_block = np.zeros((X_train_cat.shape[0], len(categories)), dtype=np.float64)
        for row_idx, value in enumerate(X_train_cat[:, col_idx].astype(str)):
            train_block[row_idx, mapping[value]] = 1.0

        test_block = np.zeros((X_test_cat.shape[0], len(categories)), dtype=np.float64)
        for row_idx, value in enumerate(X_test_cat[:, col_idx].astype(str)):
            mapped_idx = mapping.get(value)
            if mapped_idx is not None:
                test_block[row_idx, mapped_idx] = 1.0

        train_blocks.append(train_block)
        test_blocks.append(test_block)

    return np.concatenate(train_blocks, axis=1), np.concatenate(test_blocks, axis=1)


def _prepare_features(X_train_raw, X_test_raw):
    numeric_cols, categorical_cols = _infer_numeric_and_categorical_columns(X_train_raw)
    train_blocks = []
    test_blocks = []

    if numeric_cols:
        train_numeric = X_train_raw[:, numeric_cols].astype(np.float64)
        test_numeric = X_test_raw[:, numeric_cols].astype(np.float64)
        train_blocks.append(train_numeric)
        test_blocks.append(test_numeric)

    if categorical_cols:
        train_categorical, test_categorical = _one_hot_encode_train_test(
            X_train_raw[:, categorical_cols],
            X_test_raw[:, categorical_cols],
        )
        train_blocks.append(train_categorical)
        test_blocks.append(test_categorical)

    X_train = np.concatenate(train_blocks, axis=1)
    X_test = np.concatenate(test_blocks, axis=1)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    X_train = ((X_train - mean) / std).astype(np.float64)
    X_test = ((X_test - mean) / std).astype(np.float64)
    return X_train, X_test


def _stratified_subset_indices(y, subset_size, seed):
    rng = np.random.default_rng(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    expected = counts * (subset_size / len(y))
    per_class = np.floor(expected).astype(int)
    per_class = np.maximum(per_class, 1)

    while per_class.sum() > subset_size:
        candidates = np.where(per_class > 1)[0]
        if len(candidates) == 0:
            break
        candidate = candidates[np.argmax(expected[candidates] - per_class[candidates])]
        per_class[candidate] -= 1

    while per_class.sum() < subset_size:
        candidate = np.argmax(expected - per_class)
        if per_class[candidate] < counts[candidate]:
            per_class[candidate] += 1
        else:
            break

    indices = []
    for class_value, class_size in zip(unique_classes, per_class):
        class_indices = np.flatnonzero(y == class_value)
        rng.shuffle(class_indices)
        indices.append(class_indices[:class_size])

    subset_indices = np.concatenate(indices)
    rng.shuffle(subset_indices)
    return subset_indices


def _subsample_stratified(X, y, max_size, seed):
    if max_size is None or max_size <= 0 or len(y) <= max_size:
        return X, y

    subset_indices = _stratified_subset_indices(y, max_size, seed)
    return X[subset_indices], y[subset_indices]


def _stratified_train_test_split(X, y, test_size, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for class_value in np.unique(y):
        class_indices = np.flatnonzero(y == class_value)
        rng.shuffle(class_indices)
        n_test = int(round(len(class_indices) * test_size))
        n_test = min(max(n_test, 1), len(class_indices) - 1)
        test_indices.append(class_indices[:n_test])
        train_indices.append(class_indices[n_test:])

    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def _encode_binary_labels(labels, positive_label=None):
    labels = np.asarray(labels).reshape(-1)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    if len(unique_labels) != 2:
        raise ValueError("Binary label encoding requires exactly two distinct labels.")

    if positive_label is None:
        positive_label = unique_labels[-1]
    y = np.array([1 if label == positive_label else 0 for label in labels], dtype=np.int32)
    return y, positive_label


def _download_dataset(url, destination_path):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if not os.path.exists(destination_path):
        urllib.request.urlretrieve(url, destination_path)
    return destination_path


def _load_text_table(path, delimiter=","):
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    return np.genfromtxt(io.StringIO(content), delimiter=delimiter, dtype=str)


def _load_breast_cancer(args):
    path = _download_dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        os.path.join(args.dataset_cache_dir, "wdbc.data"),
    )
    table = _load_text_table(path, delimiter=",")
    X = table[:, 2:].astype(np.float64)
    y, positive_label = _encode_binary_labels(table[:, 1], positive_label="M")
    return X, y, "Breast Cancer Wisconsin", positive_label


def _load_ionosphere(args):
    path = _download_dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        os.path.join(args.dataset_cache_dir, "ionosphere.data"),
    )
    table = _load_text_table(path, delimiter=",")
    X = table[:, :-1].astype(np.float64)
    y, positive_label = _encode_binary_labels(table[:, -1], positive_label="g")
    return X, y, "Ionosphere", positive_label


def _load_german_credit(args):
    path = _download_dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        os.path.join(args.dataset_cache_dir, "german.data"),
    )
    table = _load_text_table(path, delimiter=None)
    X = table[:, :-1]
    y, positive_label = _encode_binary_labels(table[:, -1], positive_label="2")
    return X, y, "German Credit", positive_label


def _load_covtype_binary(args):
    path = _download_dataset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
        os.path.join(args.dataset_cache_dir, "covtype.data.gz"),
    )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        table = np.genfromtxt(handle, delimiter=",", dtype=np.float64)

    X = table[:, :-1]
    y_raw = table[:, -1].astype(np.int32)
    if args.covtype_variant == "binary12":
        mask = np.isin(y_raw, [1, 2])
        X = X[mask]
        y = (y_raw[mask] == 2).astype(np.int32)
        positive_name = "class_2"
    else:
        y = (y_raw == args.covtype_positive_label).astype(np.int32)
        positive_name = f"class_{args.covtype_positive_label}"
    return np.asarray(X), np.asarray(y), positive_name


def load_blr_dataset(args):
    if args.dataset == "breast_cancer":
        X, y, dataset_name, positive_label = _load_breast_cancer(args)
    elif args.dataset == "ionosphere":
        X, y, dataset_name, positive_label = _load_ionosphere(args)
    elif args.dataset == "german_credit":
        X, y, dataset_name, positive_label = _load_german_credit(args)
    elif args.dataset == "covtype":
        X, y, positive_label = _load_covtype_binary(args)
        dataset_name = "Covertype"
    else:
        raise ValueError(f"Unsupported BLR dataset: {args.dataset}")

    X = np.asarray(X)
    y = np.asarray(y).astype(np.int32).reshape(-1)
    if np.unique(y).size != 2:
        raise ValueError(f"Dataset '{args.dataset}' is not binary after preprocessing.")

    X_train_raw, X_test_raw, y_train, y_test = _stratified_train_test_split(X, y, args.test_size, args.seed)

    X_train_raw, y_train = _subsample_stratified(
        X_train_raw,
        y_train,
        args.max_train_size,
        args.seed,
    )
    X_test_raw, y_test = _subsample_stratified(
        X_test_raw,
        y_test,
        args.max_test_size,
        args.seed + 1,
    )

    X_train, X_test = _prepare_features(X_train_raw, X_test_raw)
    metadata = {
        "dataset_name": dataset_name,
        "positive_label": str(positive_label),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "feature_dim": int(X_train.shape[1]),
    }
    return X_train, X_test, y_train, y_test, metadata
