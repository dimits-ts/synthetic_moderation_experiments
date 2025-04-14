import re
import shutil

import pandas as pd
import numpy as np

from . import constants


def main_dataset() -> pd.DataFrame:
    shutil.unpack_archive("../data/dataset.zip", "../data")
    full_df = pd.read_csv("../data/dataset.csv", encoding="utf8")

    full_df.conv_variant = full_df.conv_variant.map(
        constants.MODERATION_STRATEGY_MAP
    )
    full_df.model = full_df.model.map(constants.MODEL_MAP)
    full_df = _format_main_dataset(full_df, min_message_len=3)
    full_df = full_df.rename(constants.METRIC_MAP, axis=1)
    return full_df


def _format_main_dataset(
    df: pd.DataFrame, min_message_len: int
) -> pd.DataFrame:
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
    return df


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
