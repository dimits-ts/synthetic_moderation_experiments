import re

import pandas as pd
import numpy as np


def format_dataset(df: pd.DataFrame, min_message_len: int) -> pd.DataFrame:
    df = df.astype(str)

    # Extract all annotations from the 'annotation' column
    annotations = df["annotation"].apply(get_annotations)

    # Convert each annotation dictionary into separate columns
    annotations_df = pd.json_normalize(annotations)

    # Concatenate the new columns with the original dataframe
    df = pd.concat([df, annotations_df], axis=1)
    df = df[(df.toxicity != -1) | (df.argumentquality != -1)]

    df.message_order = df.message_order.astype(int)

    # Process other columns as needed
    df.is_moderator = (df.is_moderator == "True").astype(bool)
    df["intent"] = df.user_prompt.apply(get_user_intent).astype(str)
    df.intent = np.where(df.is_moderator, "Moderator", df.intent).astype(str)

    df["not_intervened"] = (
        df.is_moderator & df.message.apply(lambda x: len(x.strip()) < min_message_len)
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


def get_annotations(annot_str: str) -> dict:
    """Extracts all key-value pairs from the annotation string into a dictionary."""
    try:
        annot_str = str(annot_str).lower()
        # Regex to match key-value pairs of the form type=value
        pattern = r"(\w+)=([-\d\.]+)"
        matches = re.findall(pattern, annot_str)
        return {
            key: float(value) if "." in value else int(value) for key, value in matches
        }
    except Exception as e:
        return {}


def get_user_intent(prompt: str) -> str:
    prompt = prompt.lower()

    if "community" in prompt:
        return "Community-oriented"
    elif "troll" in prompt:
        return "Troll"
    elif "special_instructions: ," in prompt:
        return "Neutral"
    else:
        return "Unknown"
