from pathlib import Path

import pandas as pd


INPUT_PATH = Path("raw/conversations-gone-awry-cmv-corpus/utterances.jsonl")
OUTPUT_PATH = Path("final/cmv_awry2.csv")
PERCENTILE_ESCALATION = 60


def main():
    df = pd.read_json(INPUT_PATH, lines=True)

    deleted_comments = df.text == "[deleted]"
    print(f"Removed {len(deleted_comments)} deleted comments")
    df = df[~deleted_comments]

    # keep pure derailment as note
    df.meta = df.meta.apply(lambda _dict: f"Derailment={_dict['score']}")
    df = df.rename(
        columns={
            "conversation_id": "conv_id",
            "id": "message_id",
            "speaker": "user",
            "reply-to": "reply_to",
            "text": "text",
            "meta": "notes",
        }
    )

    df["escalated"] = df.notes.apply(lambda x: int(x.split("=")[1]))
    threshold = df["escalated"].quantile(PERCENTILE_ESCALATION / 100)
    df["escalated"] = df["escalated"] > threshold
    df = df.loc[
        :,
        [
            "conv_id",
            "message_id",
            "reply_to",
            "user",
            "escalated",
            "text",
        ],
    ]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
