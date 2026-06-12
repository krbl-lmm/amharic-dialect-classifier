from pathlib import Path
import pandas as pd


def read_dialect_folder(folder_path):
    parquet_files = sorted(Path(folder_path).glob("*.parquet"))

    dfs = []

    for file in parquet_files:
        dfs.append(pd.read_parquet(file))

    return pd.concat(dfs, ignore_index=True)


def sample_dialect(df, n_samples=200):
    return df.sample(
        n=min(n_samples, len(df)),
        random_state=101
    )


def load_all_datasets(data_dir="data", dev_mode=True):
    dialects = {
        "addis_ababa": "addis_ababa",
        "gonder": "gonder",
        "gojjam": "gojjam",
        "shewa": "shewa",
        "wello": "wello",
    }

    all_dfs = []

    for folder in dialects.values():
        df = read_dialect_folder(
            Path(data_dir) / folder
        )

        if dev_mode:
            df = sample_dialect(df, 200)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)