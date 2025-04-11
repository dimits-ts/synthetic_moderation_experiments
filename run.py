"""
Imports the configuration file, sets up logging, and directs calls for
 discussion generation, annotation, and dataset export.
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import yaml

import syndisco.experiments
import syndisco.backend
import syndisco.postprocessing
import syndisco.util.model_util
import syndisco.util.file_util
import syndisco.util.logging_util

logger = logging.getLogger(Path(__file__).name)


def main():
    """
    Run synthetic discussion generation, annotation, and dataset export.

    This function parses the configuration file, sets up logging, and performs
    tasks based on the actions specified in the configuration. Actions include
    generating synthetic discussions, creating annotations, and exporting the
    dataset.
    """
    # Set up argument parser for config file path
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations"
    )
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_file, "r", encoding="utf8") as file:
        yaml_data = yaml.safe_load(file)

    model_manager = syndisco.util.model_util.ModelManager(
        model_path=yaml_data["model"]["model_path"],
        model_pseudoname=yaml_data["model"]["model_pseudoname"],
        max_new_tokens=yaml_data["model"]["max_tokens"],
        disallowed_strings=yaml_data["model"]["disallowed_strings"],
    )

    generate_discussions = yaml_data["actions"]["generate_discussions"]
    generate_annotations = yaml_data["actions"]["generate_annotations"]
    export_dataset = yaml_data["actions"]["export_dataset"]

    setup_logging(logging_config=yaml_data["logging"])
    validate_actions(
        generate_discussions=generate_discussions,
        generate_annotations=generate_annotations,
        export_dataset=export_dataset,
    )

    if generate_discussions:
        discussion_exp = create_discussion_experiment(
            llm=model_manager.get(), discussion_config=yaml_data["discussions"]
        )
        run_discussion_experiment(
            experiment=discussion_exp,
            output_dir=Path(yaml_data["discussions"]["files"]["output_dir"]),
        )

    if generate_annotations:
        ann_exp = create_annotation_experiment(
            llm=model_manager.get(), annotation_config=yaml_data["annotation"]
        )
        run_annotation_experiment(
            ann_exp,
            discussions_dir=Path(
                yaml_data["discussions"]["files"]["output_dir"]
            ),
            output_dir=Path(yaml_data["annotation"]["files"]["output_dir"]),
        )

    if export_dataset:
        dataset_to_csv(yaml_data["dataset_export"])


def setup_logging(logging_config: dict) -> None:
    syndisco.util.logging_util.logging_setup(
        print_to_terminal=logging_config["print_to_terminal"],
        write_to_file=logging_config["write_to_file"],
        logs_dir=Path(logging_config["logs_dir"]),
        level=logging_config["level"],
        use_colors=True,
        log_warnings=True,
    )


def validate_actions(
    generate_discussions, generate_annotations, export_dataset
) -> None:
    if (
        not generate_discussions
        and not generate_annotations
        and not export_dataset
    ):
        logger.warning(
            "All procedures have been disabled for this run. Exiting..."
        )
        sys.exit(0)
    else:
        if not generate_discussions:
            logger.warning("Synthetic discussion generation disabled.")
        if not generate_annotations:
            logger.warning("Synthetic annotation disabled.")
        if not export_dataset:
            logger.warning("Dataset export to CSV disabled.")


def create_discussion_experiment(
    llm, discussion_config: dict
) -> syndisco.experiments.DiscussionExperiment:
    topics = syndisco.util.file_util.read_files_from_directory(
        discussion_config["files"]["topics_dir"]
    )

    users = get_users(llm, discussion_config)
    moderator = get_mod(llm, discussion_config)
    next_turn_manager = get_turn_manager(
        turn_manager_type=discussion_config["turn_taking"][
            "turn_manager_type"
        ],
        p_respond=discussion_config["turn_taking"]["respond_probability"],
    )

    return syndisco.experiments.DiscussionExperiment(
        seed_opinions=topics,
        users=users,
        moderator=moderator,
        next_turn_manager=next_turn_manager,
        num_turns=discussion_config["turn_taking"]["num_turns"],
        num_active_users=discussion_config["experiment_variables"][
            "num_users"
        ],
        num_discussions=discussion_config["experiment_variables"][
            "num_experiments"
        ],
    )


def get_users(
    llm: syndisco.backend.model.BaseModel, discussion_config: dict
) -> list[syndisco.backend.actors.LLMActor]:
    return syndisco.backend.actors.create_users_from_file(
        llm,
        persona_path=Path(discussion_config["files"]["user_persona_path"]),
        instruction_path=Path(
            discussion_config["files"]["user_instructions_path"]
        ),
        context=discussion_config["experiment_variables"]["context_prompt"],
        actor_type=syndisco.backend.actors.ActorType.USER,
    )


def get_mod(
    llm: syndisco.backend.model.BaseModel, discussion_config: dict
) -> syndisco.backend.actors.LLMActor | None:
    if discussion_config["experiment_variables"]["include_mod"]:
        mod_instructions = syndisco.util.file_util.read_file(
            discussion_config["files"]["mod_instructions_path"]
        )
        moderator = syndisco.backend.actors.create_users(
            llm,
            usernames=["moderator"],
            attributes=[
                discussion_config["experiment_variables"][
                    "moderator_attributes"
                ]
            ],
            context=discussion_config["experiment_variables"][
                "context_prompt"
            ],
            actor_type=syndisco.backend.actors.ActorType.USER,
            instructions=mod_instructions,
        )[0]
    else:
        moderator = None
    return moderator


def get_turn_manager(
    turn_manager_type: str, p_respond: float
) -> syndisco.backend.turn_manager.TurnManager:
    if turn_manager_type == "random_weighted":
        return syndisco.backend.turn_manager.RandomWeighted(
            p_respond=p_respond
        )
    else:
        return syndisco.backend.turn_manager.RoundRobin()


def run_discussion_experiment(
    experiment: syndisco.experiments.DiscussionExperiment, output_dir: Path
) -> None:
    logger.info("Starting synthetic discussion experiments...")
    experiment.begin(discussions_output_dir=output_dir)
    logger.info("Finished synthetic discussion experiments.")


def create_annotation_experiment(
    llm, annotation_config: dict
) -> syndisco.experiments.AnnotationExperiment:
    annotators = syndisco.backend.actors.create_users_from_file(
        llm,
        persona_path=Path(
            annotation_config["files"]["annotator_persona_path"]
        ),
        instruction_path=Path(annotation_config["files"]["instruction_path"]),
        context="You are a human annotator",
        actor_type=syndisco.backend.actors.ActorType.ANNOTATOR,
    )

    return syndisco.experiments.AnnotationExperiment(
        annotators=annotators,
        history_ctx_len=annotation_config["experiment_variables"][
            "history_ctx_len"
        ],
        include_mod_comments=annotation_config["experiment_variables"][
            "include_mod_comments"
        ],
    )


def run_annotation_experiment(
    annotation_experiment: syndisco.experiments.AnnotationExperiment,
    discussions_dir: Path,
    output_dir: Path,
) -> None:
    logger.info("Starting synthetic annotation...")
    annotation_experiment.begin(discussions_dir, output_dir)
    logger.info("Finished synthetic annotation.")


def dataset_to_csv(export_config) -> None:
    conv_dir = Path(export_config["discussion_root_dir"])
    annot_dir = Path(export_config["annotation_root_dir"])
    export_path = Path(export_config["export_path"])

    df = _create_dataset(conv_dir=conv_dir, annot_dir=annot_dir)
    _export_dataset(df=df, output_path=export_path)
    logger.info(f"Dataset exported to {export_path}")


def _create_dataset(conv_dir: Path, annot_dir: Path) -> pd.DataFrame:
    """
    Create a combined dataset from conversation and annotation files.

    :param conv_dir: Directory containing conversation data files.
    :type conv_dir: Path
    :param annot_dir: Directory containing annotation data files.
    :type annot_dir: Path
    :return: A combined DataFrame containing conversation and annotation data.
    :rtype: pd.DataFrame
    """
    conv_df = syndisco.postprocessing.import_discussions(conv_dir)
    conv_df = conv_df.rename({"id": "conv_id"}, axis=1)
    annot_df = syndisco.postprocessing.import_annotations(annot_dir)

    full_df = pd.merge(
        left=conv_df,
        right=annot_df,
        on=["conv_id", "message"],
        how="left",
        suffixes=["_conv", "_annot"],  # type: ignore
    )
    del full_df["index_annot"]
    del full_df["index_conv"]

    return full_df


def _export_dataset(df: pd.DataFrame, output_path: Path):
    """
    Export the dataset to a CSV file.

    :param df: The dataset to export.
    :type df: pd.DataFrame
    :param output_path: The path where the exported dataset will be saved.
    :type output_path: Path
    """
    syndisco.util.file_util.ensure_parent_directories_exist(output_path)
    df.to_csv(
        path_or_buf=output_path, encoding="utf8", mode="w+"
    )  # overwrite previous dataset


if __name__ == "__main__":
    main()
