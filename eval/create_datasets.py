from pathlib import Path
import argparse

from tqdm.auto import tqdm
import pandas as pd

ABLATION_TAGS = ["nomod", "notrolls", "noinstr", "nosdbs"]


def get_ablation_dataset(discussions_output_root: Path) -> pd.DataFrame:
    discussion_df_ls = []
    for discussion_file in tqdm(list(discussions_output_root.rglob("*.csv"))):
        full_tag = discussion_file.parent.parent.name
        tags = full_tag.split("_")
        final_tag = tags[-1]

        if final_tag in ABLATION_TAGS:
            discussion_df = pd.read_csv(discussion_file)
            discussion_df["ablation"] = final_tag
            discussion_df_ls.append(discussion_df)
    return pd.concat(discussion_df_ls, ignore_index=True)


def get_normal_dataset(discussions_output_root: Path) -> pd.DataFrame:
    discussion_df_ls = []

    for discussion_file in tqdm(list(discussions_output_root.rglob("*.csv"))):
        full_tag = discussion_file.parent.parent.name
        tags = full_tag.split("_")
        final_tag = tags[-1]

        # skip ablation tags
        if final_tag not in ABLATION_TAGS:
            discussion_df = pd.read_csv(discussion_file)

            # add each tag as a separate column
            for i, tag in enumerate(tags):
                discussion_df[f"tag_{i}"] = tag

            # also add full tag and final tag if useful
            discussion_df["full_tag"] = full_tag
            discussion_df["final_tag"] = final_tag

            discussion_df_ls.append(discussion_df)

    return pd.concat(discussion_df_ls, ignore_index=True)



def main(discussions_output_root: Path, dataset_output_dir: Path):
    get_ablation_dataset(discussions_output_root).to_csv(
        dataset_output_dir / "ablation.csv"
    )
    get_normal_dataset(discussions_output_root).to_csv(
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
