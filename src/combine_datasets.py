from pathlib import Path

import pandas as pd

DISCUSSIONS_DIR = Path(".")

ablation_dfs = []
main_dfs = []

for directory in DISCUSSIONS_DIR.iterdir():
    dirname = directory.name
    if "nomod" in str(dirname) or "notrolls" in str(dirname):
        ablation_dfs.append(pd.)