
# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import json
import time
import requests
import argparse
from urllib.parse import urlparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BATCH_SIZE = 100
DELAY_SECS = 1.0


def get_perspective_scores(
    df: pd.DataFrame, api_key: str, out_path: Path
) -> None:
    url = (
        "https://commentanalyzer.googleapis.com/v1alpha1/"
        f"comments:analyze?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}

    # Prepare output: clear file if exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Load already scored IDs if file exists
    if out_path.exists():
        existing = pd.read_csv(out_path)
        scored_ids = set(existing["message_id"].unique())
        print(f"Loaded {len(scored_ids)} previously scored IDs.")
        wrote_header = True
    else:
        scored_ids = set()
        wrote_header = False

    batch = []
    wrote_header = False
    # handle CMV-AWRY dataset
    text_column = "message" if "message" in df.columns else "text"

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Scoring comments"
    ):
        text = row[text_column] 
        msg_id = row["message_id"]

        if text == "":
            continue

        if msg_id in scored_ids:
            continue

        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}},
            "doNotStore": True,
        }

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(data)
            )
            response.raise_for_status()
            result = response.json()
            time.sleep(DELAY_SECS)

            toxicity = _get_response(result, "TOXICITY")

            row_out = {
                "message_id": msg_id,
                "toxicity": toxicity,
                "error": None,
            }

        except requests.exceptions.RequestException as e:
            parsed = urlparse(getattr(e.request, "url", ""))
            safe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            row_out = {
                "message_id": msg_id,
                "toxicity": None,
                "error": f"{type(e).__name__}: request to {safe_url} failed",
            }

        batch.append(row_out)

        # Write batch to disk
        if len(batch) >= BATCH_SIZE:
            df_out = pd.DataFrame(batch)
            df_out.to_csv(
                out_path, mode="a", index=False, header=not wrote_header
            )
            wrote_header = True
            batch = []

    # Write remaining rows
    if batch:
        df_out = pd.DataFrame(batch)
        df_out.to_csv(out_path, mode="a", index=False, header=not wrote_header)


def _get_response(raw_res: dict, attribute: str):
    return raw_res["attributeScores"][attribute]["summaryScore"]["value"]


def main(input_csv_path: Path, output_path: Path, api_key_file: Path):
    api_key = api_key_file.read_text()
    df = pd.read_csv(input_csv_path)
    get_perspective_scores(df=df, api_key=api_key, out_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Perspective API scoring and save results to CSV."
    )
    parser.add_argument(
        "--api-key-path",
        type=str,
        required=True,
        help="Path to file containing Perspective API key",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="../pefk.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="perspective_results.csv",
        help="Output path for CSV file",
    )

    args = parser.parse_args()
    main(
        input_csv_path=Path(args.input_csv),
        output_path=Path(args.output_path),
        api_key_file=Path(args.api_key_path),
    )
