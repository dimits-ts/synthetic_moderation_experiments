from pathlib import Path
import argparse

from tqdm.auto import tqdm
import pandas as pd
import syndisco.postprocessing

ABLATION_TAGS = ["nomod", "notrolls", "noinstr", "nosdbs"]


def load_and_combine_discussions(parent_dir, source_col_name="source_dir"):
    """
    For each subdirectory in parent_dir, load discussions into a DataFrame,
    add the subdirectory name as the last column, and combine all DataFrames.

    Parameters
    ----------
    parent_dir : str or Path
        Directory containing subdirectories with JSON outputs.
    source_col_name : str
        Name of the column that will store the subdirectory name.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with an extra column indicating the source directory.
    """
    parent_dir = Path(parent_dir)
    dataframes = []

    for subdir in parent_dir.iterdir():
        if subdir.is_dir():
            df = syndisco.postprocessing.import_discussions(subdir)

            if df is not None and not df.empty:
                df[source_col_name] = subdir.name
                dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def main(discussions_output_root: Path, dataset_output_dir: Path):
    dataset_output_dir.mkdir(exist_ok=True)

    load_and_combine_discussions(discussions_output_root).to_csv(
        dataset_output_dir / "vmd.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine output discussions to main and ablation datasets."
    )
    parser.add_argument(
        "--discussions-root-dir",
        required=True,
        help="Root output dir for all exported discussions",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dir for the combined datasets",
    )
    args = parser.parse_args()
    main(
        discussions_output_root=Path(args.discussions_root_dir),
        dataset_output_dir=Path(args.output_dir),
    )
