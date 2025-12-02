from pathlib import Path

import pandas as pd


def main(discussions_output_root: Path):
    dfs = {}
    for discussion_file in discussions_output_root.rglob("*.csv"):
        full_tag = discussion_file.parent.parent.name

        if "nomod" in full_tag:
            continue
        else:
            model_name, turn_taking, mod_present = full_tag.split("_")
            mod_strat = None

        df = pd.read_csv(discussion_file)
        df["model_name"] = model_name
        df["turn_taking"] = turn_taking
        df["mod_strat"] = mod_strat
        df["mod_present"] = mod_present
        dfs.append()
    ...