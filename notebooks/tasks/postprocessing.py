import re
import shutil

import pandas as pd
import numpy as np

from . import constants


CLEAN_HTML_PATTERN = re.compile("<.*?>")


def get_main_dataset() -> pd.DataFrame:
    shutil.unpack_archive(
        constants.DATASET_DIR / "main" / "main.zip",
        constants.DATASET_DIR / "main",
    )
    full_df = pd.read_csv(
        # I have no idea wtf happened here to necessitate this
        constants.DATASET_DIR / "main" / "data" / "datasets" / "dataset.csv",
        encoding="utf8",
    )
    full_df = format_dataset(full_df, min_message_len=-1)
    return full_df


def format_dataset(df: pd.DataFrame, min_message_len: int) -> pd.DataFrame:
    df.conv_variant = df.conv_variant.map(constants.MODERATION_STRATEGY_MAP)
    df.model = df.model.map(constants.MODEL_MAP)
    df = df.astype(str)
    # Extract all annotations from the 'annotation' column
    annotations = df["annotation"].apply(_get_annotations)

    # Convert each annotation dictionary into separate columns
    annotations_df = pd.json_normalize(annotations)

    # Concatenate the new columns with the original dataframe
    df = pd.concat([df, annotations_df], axis=1)
    df = df[(df.toxicity != -1) | (df.argumentquality != -1)]

    df.message_order = df.message_order.astype(int)

    # Process other columns as needed
    df.is_moderator = (df.is_moderator == "True").astype(bool)
    df["intent"] = df.user_prompt.apply(_get_user_intent).astype(str)
    df.intent = np.where(df.is_moderator, "Moderator", df.intent).astype(str)

    df["not_intervened"] = (
        df.is_moderator
        & df.message.apply(lambda x: len(x.strip()) < min_message_len)
    ).astype(bool)

    df = df.loc[
        :,
        [
            "conv_id",
            "message_id",
            "message_order",
            "conv_variant",
            "model",
            "user",
            "user_prompt",
            "is_moderator",
            "intent",
            "message",
        ]
        + list(annotations_df.columns)
        + ["not_intervened"],
    ]
    df = df.rename(constants.METRIC_MAP, axis=1)
    return df


def get_human_df():
    human_df_dict = pd.read_excel(
        constants.DATASET_DIR / "main" / "human.xlsx",
        sheet_name=list(range(1, 11)),
    )

    human_df_ls = []
    for i in [1, 2, 3, 4, 5, 6, 8]:
        human_df = _human_forum_post(
            human_df_dict[i], post_id_idx=1, comment_id_idx=0, comment_idx=5
        )
        human_df_ls.append(human_df)

    for i in [7]:
        human_df = _human_forum_post(
            human_df_dict[i], post_id_idx=1, comment_id_idx=0, comment_idx=6
        )
        human_df_ls.append(human_df)

    for i in [8, 9]:
        human_df = _human_forum_post(
            human_df_dict[i], post_id_idx=2, comment_id_idx=0, comment_idx=5
        )
        human_df_ls.append(human_df)

    human_df = pd.concat(human_df_ls, ignore_index=True)
    return human_df


def get_ablation_df() -> pd.DataFrame:
    datasets = []
    dataset_dir = constants.DATASET_DIR / "discussion_ablation"
    for dataset_path in dataset_dir.rglob("*.csv"):
        abl_df = pd.read_csv(dataset_path)

        # each ablation feature is a dimension,
        # each factor is a value in that dimension
        df_id = dataset_path.stem.replace("abl_", "")
        feature, factor = df_id.split("_")
        abl_df[feature] = factor

        datasets.append(abl_df)

    df = pd.concat(datasets, ignore_index=True)
    return df


def _human_forum_post(
    df, post_id_idx: int, comment_id_idx: int, comment_idx: int
) -> pd.DataFrame:
    df = df.iloc[
        :,
        [post_id_idx, comment_id_idx, comment_idx],
    ]

    df = df.copy()
    df.columns = ["conv_id", "message_id", "message"]
    df.conv_id = df.conv_id.astype(str)
    df.message = df.message.astype(str)
    df.message = df.message.apply(_rem_html_tags)
    return df


# https://stackoverflow.com/a/12982689
def _rem_html_tags(raw_html: str) -> str:
    cleantext = re.sub(CLEAN_HTML_PATTERN, "", raw_html)
    return cleantext


def _get_annotations(annot_str: str) -> dict:
    """
    Extracts all key-value pairs from the annotation string into a dictionary.
    """
    try:
        annot_str = str(annot_str).lower()
        # Regex to match key-value pairs of the form type=value
        pattern = r"(\w+)=([-\d\.]+)"
        matches = re.findall(pattern, annot_str)
        return {
            key: float(value) if "." in value else int(value)
            for key, value in matches
        }
    except Exception:
        return {}


def _get_user_intent(prompt: str) -> str:
    prompt = prompt.lower()

    if "community" in prompt:
        return "Community-oriented"
    elif "troll" in prompt:
        return "Troll"
    elif "special_instructions: ," in prompt:
        return "Neutral"
    else:
        return "Unknown"
