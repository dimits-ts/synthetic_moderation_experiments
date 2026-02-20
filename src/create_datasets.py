from pathlib import Path
import argparse

import pandas as pd
import syndisco.postprocessing


def get_initialization(full_tag: str) -> str:
    return "No seeds" if "noseeds" in full_tag else "Main"


def get_strategy(full_tag: str) -> str:
    if "nomod" in full_tag:
        return "No Facilitator"
    elif "constructive" in full_tag:
        return "Constr. Comms"
    elif "erulemaking" in full_tag:
        return "E-Rulemaking"
    elif "vanilla" in full_tag:
        return "No Instructions"
    else:
        raise ValueError(f"Unknown strategy: {full_tag}")


def get_turntaking(full_tag: str) -> str:
    final_tag = full_tag.split("_")[-1]
    match final_tag:
        case "random":
            return "Random"
        case "roundrobin":
            return "Round-robin"
        case _:
            return "Response-enabled"


def get_userprompts(full_tag: str) -> str:
    final_tag = full_tag.split("_")[-1]
    match final_tag:
        case "nosdbs":
            return "No SDBs"
        case "noinstructions":
            return "Minimal instructions"
        case _:
            return "Main"


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
        Combined DataFrame with an extra column indicating
        the source directory.
    """
    parent_dir = Path(parent_dir)
    dataframes = []

    for subdir in parent_dir.iterdir():
        if subdir.is_dir():
            df = syndisco.postprocessing.import_discussions(subdir)
            tag = subdir.name
            df["strategy"] = get_strategy(tag)
            df["turn_taking"] = get_turntaking(tag)
            df["user_prompts"] = get_userprompts(tag)
            df["initialization"] = get_initialization(tag)
            dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def main(discussions_output_root: Path, output_path: Path):
    output_path.parent.mkdir(exist_ok=True)

    df = load_and_combine_discussions(discussions_output_root)
    df.model = df.model.replace(
        {
            "llama70b": "LLaMa-70B",
            "mistral24b": "Mistral-24B",
            "qwen32b": "Qwen-32B",
            "llama8b": "LLaMa-8B",
            "mistral7b": "Mistral-7B",
            "qwen7b": "Qwen-7B",
        }
    )
    df.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine output discussions to CSV."
    )
    parser.add_argument(
        "--discussions-root-dir",
        required=True,
        help="Root output dir for all exported discussions",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for the combined dataset",
    )
    args = parser.parse_args()
    main(
        discussions_output_root=Path(args.discussions_root_dir),
        output_path=Path(args.output_path),
    )
