import pandas as pd

import math
import os
import json
import re


def import_and_format_conversations(conv_dir: str) -> pd.DataFrame:
    conv_df = import_conversations(conv_dir)
    conv_df = _split_prompts(conv_df)
    conv_df = _add_moderator_exists(conv_df)

    attribute_df = _split_sdbs(conv_df, prompt_col="user_prompt")
    combined_df = pd.concat([conv_df, attribute_df], axis=1)

    return combined_df


def import_and_format_annotations(
    annot_dir: str, round: bool, sentinel_value: int
) -> pd.DataFrame:
    annot_df = import_annotations(annot_dir, round=round, sentinel_value=sentinel_value)
    annot_df = annot_df[annot_df.toxicity != sentinel_value]

    attribute_df = _split_sdbs(annot_df, prompt_col="annotator_prompt") #type: ignore
    combined_df = pd.concat([annot_df, attribute_df], axis=1)

    return combined_df


def import_conversations(conv_dir: str) -> pd.DataFrame:
    """
    Import conversation data from a directory containing JSON files and convert them to a DataFrame.

    Recursively reads all JSON files from the specified directory,
    and extracts relevant fields. It also adds metadata about the conversation variant.

    :param conv_dir: Path to the root directory containing the conversation JSON files.
    :type conv_dir: str
    :return: A DataFrame with conversation data, including the ID, user prompts, messages,
             and conversation variant.
    :rtype: pd.DataFrame

    :example:
        >>> df = import_conversations("/path/to/conversation/data")
    """
    file_paths = _files_from_dir_recursive(conv_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")
        # get name, not path of parent directory
        conv["conv_variant"] = os.path.basename(os.path.dirname(file_path))
        conv["user"] = conv.logs.apply(lambda x: x[0])
        conv["message"] = conv.logs.apply(lambda x: x[1])
        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    return full_df


def import_annotations(
    annot_dir: str, round: bool = True, sentinel_value: int = -1
) -> pd.DataFrame:
    """
    Import annotation data from a directory containing JSON files and convert them to a DataFrame.

    Recursively reads all JSON files from the specified directory, and extracts relevant fields. Also parses annotator
    attributes and toxicity values from the logs.

    :param annot_dir: Path to the directory containing the annotation JSON files.
    :type annot_dir: str
    :param round: Whether to discretize annotation values to integers. If so, NaN values will  be replaced with sentinel_value
    :type round: bool
    :param sentinel_value: The value of NaN annotation values. Used only if round_down is True
    :type sentinel_value: bool
    :return: A DataFrame with annotation data, including conversation ID, annotator prompts,
             messages, and toxicity values.
    :rtype: pd.DataFrame

    :example:
        >>> df = import_annotations("/path/to/annotation/data")
    """
    file_paths = _files_from_dir_recursive(annot_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")
        conv["message"] = conv.logs.apply(lambda x: x[0])
        conv["toxicity"] = conv.logs.apply(lambda x: x[1])
        conv["toxicity"] = conv.toxicity.apply(_extract_toxicity_value)

        if round:
            conv["toxicity"] = conv["toxicity"].apply(
                lambda x: sentinel_value if x is None or math.isnan(x) else int(x)
            )

        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    return full_df


# code adapted from https://www.geeksforgeeks.org/python-list-all-files-in-directory-and-subdirectories/
def _files_from_dir_recursive(start_path="."):
    """
    Recursively list all files in a directory and its subdirectories.

    :param start_path: The starting directory path. Defaults to the current directory.
    :type start_path: str, optional
    :return: A list of file paths.
    :rtype: list[str]

    :example:
       >>> file_paths = _files_from_dir_recursive("/path/to/data")
    """
    all_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def _split_prompts(df: pd.DataFrame) -> pd.DataFrame:
    df["user_prompt"] = df.apply(
        lambda row: _extract_user_prompt(row["user_prompts"], row["user"]),
        axis=1,
    )

    df["user_prompt"] = df.apply(
        lambda row: (
            row["moderator_prompt"]
            if row["user"] == "moderator"
            else row["user_prompt"]
        ),
        axis=1,
    )

    df["user_prompt"] = df["user_prompt"].apply(_sdb_portion)
    return df


def _add_moderator_exists(df: pd.DataFrame) -> pd.DataFrame:
    moderator_ids = set(df[df["user"] == "moderator"]["id"])
    df["moderator_exists"] = df["id"].apply(lambda x: x in moderator_ids)
    return df


# separate the prompt of the user who is speaking
def _extract_user_prompt(prompts, user):
    for prompt in prompts:
        if f"You are {user}" in prompt:
            return prompt
    return None  # If no prompt matches


def _sdb_portion(prompt: str) -> str:
    return prompt.split("Context:")[0]


def _split_sdbs(df: pd.DataFrame, prompt_col: str) -> pd.DataFrame:
    return df[prompt_col].apply(_extract_sdb_attributes).apply(pd.Series)  # type: ignore


def _extract_sdb_attributes(prompt: str):
    # mostly generated by ChatGPT
    attributes = {}

    # Extracting age based on the pattern "NN years old"
    age_match = re.search(r"(\d{2})\s+years\s+old", prompt, re.IGNORECASE)
    attributes["age"] = int(age_match.group(1)) if age_match else None

    # Extracting potential gender by searching for words like "man", "woman"
    gender_match = re.search(r"\b(man|woman|non-binary)\b", prompt, re.IGNORECASE)
    attributes["gender"] = gender_match.group(1).capitalize() if gender_match else None

    # Extracting profession based on the assumption it appears after an age or gender mention
    attributes["profession"] = (
        "unemployed" if "unemployed" in prompt.lower() else "employed"
    )

    # Extracting education level based on patterns like "with [education] education"
    education_match = re.search(
        r"\bwith\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+education\b", prompt
    )
    attributes["education"] = education_match.group(1) if education_match else None

    # Identifying if a non-heterosexual orientation exists
    non_hetero_match = re.search(r"\b(?!Heterosexual\b)[A-Z][a-z]*sexual\b", prompt)
    attributes["is_heterosexual"] = not bool(non_hetero_match)

    # Extracting intent if it follows "and" and ends the sentence
    intent_match = re.search(r"\band\s+([A-Z][a-z]+)\s+intent\b", prompt)
    attributes["intent"] = intent_match.group(1).capitalize() if intent_match else None

    # Extracting traits as any lowercase words that aren't matched by the above patterns
    # This assumes traits are adjectives or descriptive words/phrases
    matched_text = set(match.group() for match in re.finditer(r"\b[a-z]+\b", prompt))
    extracted_values = {
        str(value).lower() for key, value in attributes.items() if value
    }
    unmatched_traits = matched_text - extracted_values
    attributes["traits"] = list(unmatched_traits)

    return attributes


def _extract_toxicity_value(text: str) -> float | None:
    """
    Extract toxicity value from a given text using a regular expression.

    This function searches for the pattern "Toxicity=<number>" in the provided text and
    returns the toxicity value as a string. If no match is found, it returns None.

    :param text: The input string containing toxicity information.
    :type text: str
    :return: The extracted toxicity value, or None if no match is found.
    :rtype: float | None

    :example:
        >>> toxicity = _extract_toxicity_value("Toxicity=4.5")
        >>> print(toxicity)  # Output: "4.5"
    """
    # Regex pattern to match "Toxicity=<number>"
    pattern = r"Toxicity=(\d+\.?\d*)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None
