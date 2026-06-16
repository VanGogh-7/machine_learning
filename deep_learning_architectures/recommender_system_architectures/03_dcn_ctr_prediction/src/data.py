from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


UNK_CATEGORY = "<unk>"
MISSING_CATEGORY = "<missing>"
RAW_CRITEO_COLUMNS = [
    "label",
    *[f"I{index}" for index in range(1, 14)],
    *[f"C{index}" for index in range(1, 27)],
]
RAW_FILE_NAMES = (
    "train.txt",
    "train.tsv",
    "dac_sample.txt",
    "dac_sample.tsv",
    "criteo_train.txt",
    "criteo_train.tsv",
)


class CTRDataset(Dataset):
    def __init__(
        self,
        numerical_features: np.ndarray,
        categorical_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        self.categorical_features = torch.tensor(
            categorical_features,
            dtype=torch.long,
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.numerical_features[index],
            self.categorical_features[index],
            self.labels[index],
        )


def validate_split_ratios(valid_ratio: float, test_ratio: float) -> None:
    if valid_ratio <= 0 or test_ratio <= 0 or valid_ratio + test_ratio >= 1:
        raise ValueError(
            "valid_ratio and test_ratio must be positive and sum to less than 1."
        )


def split_dataframe(
    dataframe: pd.DataFrame,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_split_ratios(valid_ratio, test_ratio)
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_size = max(1, int(len(shuffled) * valid_ratio))
    test_size = max(1, int(len(shuffled) * test_ratio))
    if valid_size + test_size >= len(shuffled):
        raise ValueError("The processed CTR dataset is too small for the split.")

    valid_df = shuffled.iloc[:valid_size].reset_index(drop=True)
    test_df = shuffled.iloc[valid_size:valid_size + test_size].reset_index(drop=True)
    train_df = shuffled.iloc[valid_size + test_size:].reset_index(drop=True)
    return train_df, valid_df, test_df


def find_single_processed_csv(data_root: Path, expected_paths: list[Path]) -> Path | None:
    if not data_root.exists():
        return None
    csv_paths = sorted(path for path in data_root.glob("*.csv") if path.is_file())
    if len(csv_paths) == 1:
        return csv_paths[0]

    existing_expected = [path for path in expected_paths if path.is_file()]
    if len(existing_expected) == 1 and len(csv_paths) == 1:
        return existing_expected[0]
    return None


def find_raw_criteo_file(data_root: Path) -> Path | None:
    for file_name in RAW_FILE_NAMES:
        candidate = data_root / file_name
        if candidate.is_file():
            return candidate
    return None


def read_raw_criteo_file(raw_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=RAW_CRITEO_COLUMNS,
    )
    if dataframe.shape[1] != len(RAW_CRITEO_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_CRITEO_COLUMNS)} Criteo columns in {raw_path}, "
            f"but found {dataframe.shape[1]}."
        )
    return dataframe


def save_processed_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> None:
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)


def load_raw_splits(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_paths = [train_path, valid_path, test_path]
    existing_paths = [path for path in expected_paths if path.is_file()]

    if len(existing_paths) == 3:
        return tuple(pd.read_csv(path) for path in expected_paths)

    if len(existing_paths) not in {0, 1}:
        missing = [path for path in expected_paths if not path.is_file()]
        raise FileNotFoundError(
            "Only some processed CTR split files were found. Provide all three "
            "processed files, or provide one CSV that can be split.\n"
            f"Missing files: {missing}"
        )

    data_root = train_path.parent
    single_csv = (
        existing_paths[0]
        if len(existing_paths) == 1
        else find_single_processed_csv(data_root, expected_paths)
    )
    if single_csv is not None:
        dataframe = pd.read_csv(single_csv)
        return split_dataframe(dataframe, valid_ratio, test_ratio, seed)

    raw_path = find_raw_criteo_file(data_root)
    if raw_path is not None:
        dataframe = read_raw_criteo_file(raw_path)
        train_df, valid_df, test_df = split_dataframe(
            dataframe,
            valid_ratio,
            test_ratio,
            seed,
        )
        save_processed_splits(
            train_df,
            valid_df,
            test_df,
            train_path,
            valid_path,
            test_path,
        )
        return train_df, valid_df, test_df

    expected = "\n".join(f"  - {path}" for path in expected_paths)
    raise FileNotFoundError(
        "Processed Criteo-style CTR files were not found.\n"
        "Place train.csv, valid.csv, and test.csv under the centralized dataset "
        "directory, or place one processed CSV there so it can be split.\n"
        "The project does not create a toy dataset.\n"
        f"Expected files:\n{expected}"
    )


def resolve_feature_names(
    train_df: pd.DataFrame,
    label_col: str,
    numerical_feature_names: tuple[str, ...],
    categorical_feature_names: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    if label_col not in train_df.columns:
        raise ValueError(f"Missing required label column: {label_col}")

    columns = [column for column in train_df.columns if column != label_col]
    numerical_names = [
        column for column in numerical_feature_names if column in columns
    ]
    categorical_names = [
        column for column in categorical_feature_names if column in columns
    ]
    assigned = set(numerical_names + categorical_names)

    for column in columns:
        if column in assigned:
            continue
        if pd.api.types.is_numeric_dtype(train_df[column]):
            numerical_names.append(column)
        else:
            categorical_names.append(column)

    if not numerical_names and not categorical_names:
        raise ValueError("No usable CTR feature columns were found.")
    return numerical_names, categorical_names


def validate_columns(
    dataframe: pd.DataFrame,
    feature_names: list[str],
    label_col: str,
    split_name: str,
) -> None:
    required = [label_col, *feature_names]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {missing}")


def fit_feature_metadata(
    train_df: pd.DataFrame,
    numerical_feature_names: list[str],
    categorical_feature_names: list[str],
    label_col: str,
    min_category_freq: int,
) -> dict[str, Any]:
    numerical_stats = {}
    for feature_name in numerical_feature_names:
        values = pd.to_numeric(train_df[feature_name], errors="coerce").fillna(0.0)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        numerical_stats[feature_name] = {
            "mean": mean,
            "std": std if std > 0 else 1.0,
        }

    categorical_vocabularies = {}
    for feature_name in categorical_feature_names:
        values = train_df[feature_name].fillna(MISSING_CATEGORY).astype(str)
        counts = values.value_counts()
        frequent_values = sorted(
            value
            for value, count in counts.items()
            if count >= min_category_freq
            and value not in {UNK_CATEGORY, MISSING_CATEGORY}
        )
        vocabulary = {UNK_CATEGORY: 0, MISSING_CATEGORY: 1}
        vocabulary.update(
            {value: index + 2 for index, value in enumerate(frequent_values)}
        )
        categorical_vocabularies[feature_name] = vocabulary

    return {
        "label_col": label_col,
        "label_mapping": {"not_clicked": 0, "clicked": 1},
        "numerical_feature_names": numerical_feature_names,
        "categorical_feature_names": categorical_feature_names,
        "numerical_stats": numerical_stats,
        "categorical_vocabularies": categorical_vocabularies,
        "category_sizes": [
            len(categorical_vocabularies[name])
            for name in categorical_feature_names
        ],
    }


def transform_dataframe(
    dataframe: pd.DataFrame,
    feature_metadata: dict[str, Any],
    require_label: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    numerical_arrays = []
    for feature_name in feature_metadata["numerical_feature_names"]:
        stats = feature_metadata["numerical_stats"][feature_name]
        values = pd.to_numeric(dataframe[feature_name], errors="coerce").fillna(0.0)
        normalized = (values.to_numpy(dtype=np.float32) - stats["mean"]) / stats[
            "std"
        ]
        numerical_arrays.append(normalized)

    numerical_features = (
        np.stack(numerical_arrays, axis=1)
        if numerical_arrays
        else np.empty((len(dataframe), 0), dtype=np.float32)
    )

    categorical_arrays = []
    for feature_name in feature_metadata["categorical_feature_names"]:
        vocabulary = feature_metadata["categorical_vocabularies"][feature_name]
        values = dataframe[feature_name].fillna(MISSING_CATEGORY).astype(str)
        category_ids = values.map(lambda value: vocabulary.get(value, 0))
        categorical_arrays.append(category_ids.to_numpy(dtype=np.int64))

    categorical_features = (
        np.stack(categorical_arrays, axis=1)
        if categorical_arrays
        else np.empty((len(dataframe), 0), dtype=np.int64)
    )

    labels = None
    label_col = feature_metadata["label_col"]
    if require_label:
        labels = pd.to_numeric(dataframe[label_col], errors="raise").to_numpy(
            dtype=np.float32
        )
        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("CTR labels must be binary values: 0 or 1.")
    return numerical_features, categorical_features, labels


def build_datasets(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    numerical_feature_names: tuple[str, ...],
    categorical_feature_names: tuple[str, ...],
    label_col: str,
    min_category_freq: int,
) -> tuple[CTRDataset, CTRDataset, CTRDataset, dict[str, Any]]:
    train_df, valid_df, test_df = load_raw_splits(
        train_path,
        valid_path,
        test_path,
        valid_ratio,
        test_ratio,
        seed,
    )
    numerical_names, categorical_names = resolve_feature_names(
        train_df,
        label_col,
        numerical_feature_names,
        categorical_feature_names,
    )
    all_features = [*numerical_names, *categorical_names]
    validate_columns(train_df, all_features, label_col, "Training")
    validate_columns(valid_df, all_features, label_col, "Validation")
    validate_columns(test_df, all_features, label_col, "Test")

    feature_metadata = fit_feature_metadata(
        train_df,
        numerical_names,
        categorical_names,
        label_col,
        min_category_freq,
    )
    transformed_splits = [
        transform_dataframe(dataframe, feature_metadata)
        for dataframe in (train_df, valid_df, test_df)
    ]
    datasets = [
        CTRDataset(numerical, categorical, labels)
        for numerical, categorical, labels in transformed_splits
        if labels is not None
    ]
    return datasets[0], datasets[1], datasets[2], feature_metadata


def build_dataloaders(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    batch_size: int,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
    numerical_feature_names: tuple[str, ...],
    categorical_feature_names: tuple[str, ...],
    label_col: str,
    min_category_freq: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    train_dataset, valid_dataset, test_dataset, metadata = build_datasets(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
        numerical_feature_names=numerical_feature_names,
        categorical_feature_names=categorical_feature_names,
        label_col=label_col,
        min_category_freq=min_category_freq,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, metadata
