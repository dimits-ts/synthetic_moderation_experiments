import argparse
import logging
import yaml
import random
import json
from pathlib import Path

import pandas as pd
import randomname
import syndisco.model
import syndisco.actors
import syndisco.experiments
import syndisco.logging_util
import syndisco.turn_manager
import syndisco.postprocessing

TROLL_CHANCE = 0.3


def main(
    config_file_path: Path,
    user_model_url: str,
    user_model_name: str,
    mod_model_url: str,
    mod_model_name: str,
    turn_manager_type: str,
    num_experiments: int,
    mod_active: bool,
    mod_strategy_path: Path,
    user_instruction_path: Path,
    user_persona_path: Path,
    output_dir: Path,
    trolls_active: bool,
) -> None:
    with open(config_file_path, "r", encoding="utf8") as file:
        yaml_data = yaml.safe_load(file)

    json_output_dir = output_dir / "raw"
    dataset_export_dir = output_dir / "datasets"

    run_logger_name = (
        f"run.{user_model_name}.{turn_manager_type}"
        f".mod{'On' if mod_active else 'Off'}"
        f".trolls{'On' if trolls_active else 'Off'}"
    )
    logger = logging.getLogger(run_logger_name)
    setup_logging(logging_config=yaml_data["logging"])

    already_executed = output_dir.is_dir() and any(
        item.is_file() for item in output_dir.rglob("*")
    )

    logger.info(
        "================================================\n"
        f"{json.dumps(vars(args), indent=2)}"
        "\n================================================"
    )

    if already_executed:
        logger.info(
            f'Skipping experiment since "{output_dir.resolve()}" already exists.'
        )
        return

    try:
        user_model = syndisco.model.TransformersModel(
            model_path=user_model_url,
            name=user_model_name,
            remove_string_list=[],
            max_out_tokens=yaml_data["discussion_model"]["max_tokens"],
        )

        mod_model = syndisco.model.TransformersModel(
            model_path=mod_model_url,
            name=mod_model_name,
            remove_string_list=[],
            max_out_tokens=yaml_data["discussion_model"]["max_tokens"],
        )

        discussion_exp = create_discussion_experiment(
            llm=user_model,
            mod_llm=mod_model,
            discussion_config=yaml_data["discussions"],
            include_mod=mod_active,
            mod_strategy_path=mod_strategy_path,
            user_instruction_path=user_instruction_path,
            user_persona_path=user_persona_path,
            turn_manager_type=turn_manager_type,
            trolls_active=trolls_active,
            num_experiments=num_experiments,
        )

        run_discussion_experiment(
            experiment=discussion_exp,
            output_dir=json_output_dir,
            logger=logger,
        )

        export_dataset(
            json_output_dir=json_output_dir,
            dataset_export_dir=dataset_export_dir,
            dataset_name="discussion.csv",
            logger=logger,
        )

    except Exception as e:
        logger.critical("Error while running experiment " + run_logger_name)
        logger.exception(e)


def export_dataset(
    json_output_dir: Path,
    dataset_export_dir: Path,
    dataset_name: str,
    logger: logging.Logger,
) -> None:
    dataset_export_dir.mkdir(exist_ok=True, parents=True)
    export_path = dataset_export_dir / dataset_name
    conv_df = syndisco.postprocessing.import_discussions(json_output_dir)
    conv_df.to_csv(path_or_buf=export_path, encoding="utf8", mode="w")
    logger.info(f"Dataset exported to {export_path}")


def setup_logging(logging_config: dict) -> None:
    syndisco.logging_util.logging_setup(
        print_to_terminal=logging_config["print_to_terminal"],
        write_to_file=logging_config["write_to_file"],
        logs_dir=Path(logging_config["logs_dir"]),
        level=logging_config["level"],
        use_colors=True,
        log_warnings=True,
    )


def create_discussion_experiment(
    llm,
    mod_llm,
    discussion_config: dict,
    include_mod: bool,
    mod_strategy_path: Path,
    user_instruction_path: Path,
    user_persona_path: Path,
    turn_manager_type: str,
    trolls_active: bool,
    num_experiments: int,
) -> syndisco.experiments.DiscussionExperiment:

    context = discussion_config["experiment_variables"]["context_prompt"]

    topics = get_topics(
        topics_path=Path(discussion_config["files"]["topics_path"])
    )

    users = get_users(
        model=llm,
        context=context,
        vanilla_instructions=user_instruction_path.read_text(),
        troll_instructions=Path(
            discussion_config["files"]["troll_instructions_path"]
        ).read_text(),
        trolls_active=trolls_active,
        troll_chance=TROLL_CHANCE,
        persona_file_path=user_persona_path,
    )

    if not include_mod:
        moderator = None
    else:
        mod_persona = syndisco.actors.Persona(
            username="moderator",
            age=30,
            sex="male",
            sexual_orientation="unknown",
            demographic_group="unknown",
            current_employment="moderator",
            education_level="bachelors",
            special_instructions="",
            personality_characteristics=["fair"],
        )
        moderator = get_mod(
            model=mod_llm,
            mod_persona=mod_persona,
            context=context,
            mod_instructions_path=mod_strategy_path,
        )

    next_turn_manager = get_turn_manager(
        turn_manager_type=turn_manager_type,
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
        num_discussions=num_experiments,
    )


def get_topics(topics_path: Path) -> list[list[str]]:
    df = pd.read_csv(topics_path)

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

    df = df.sort_values(["conv_id"]).reset_index(drop=True)

    # Create a positional index within each conversation
    df["pos"] = df.groupby("conv_id").cumcount()

    # Filter to escalated rows that are *not* the first comment
    escalated_df = df[(df.escalated) & (df.pos > 0)]

    # Sample one per conversation (no deprecation warning)
    sampled = escalated_df.groupby("conv_id", group_keys=False).sample(n=1)

    results = []
    for _, row in sampled.iterrows():
        conv_id = row["conv_id"]
        pos = row["pos"]

        # previous comment = same conv_id, pos - 1
        prev = df[(df.conv_id == conv_id) & (df.pos == pos - 1)]
        prev_text = prev["text"].iloc[0]

        curr_text = row["text"]
        results.append([prev_text, curr_text])

    return results


def get_users(
    model: syndisco.model.BaseModel,
    context: str,
    vanilla_instructions: str,
    troll_instructions: str,
    trolls_active: bool,
    troll_chance: float,
    persona_file_path: Path,
) -> list[syndisco.actors.Actor]:
    personas = syndisco.actors.Persona.from_json_file(
        file_path=persona_file_path
    )

    actors = []
    for persona in personas:
        persona.username = randomname.get_name()
        if not trolls_active:
            instructions = vanilla_instructions
        else:
            instructions = (
                vanilla_instructions
                if random.random() >= troll_chance
                else troll_instructions
            )

        actor = syndisco.actors.Actor(
            model=model,
            persona=persona,
            context=context,
            instructions=instructions,
            actor_type=syndisco.actors.ActorType.USER,
        )
        actors.append(actor)
    return actors


def get_mod(
    model: syndisco.model.BaseModel,
    mod_instructions_path: Path,
    mod_persona: syndisco.actors.Persona,
    context: str,
) -> syndisco.actors.Actor:
    mod_instructions = mod_instructions_path.read_text()
    mod = syndisco.actors.Actor(
        model=model,
        persona=mod_persona,
        context=context,
        instructions=mod_instructions,
        actor_type=syndisco.actors.ActorType.USER,
    )
    return mod


def get_turn_manager(
    turn_manager_type: str, p_respond: float
) -> syndisco.turn_manager.TurnManager:
    match (turn_manager_type):
        case "random-weighted":
            return syndisco.turn_manager.RandomWeighted(p_respond=p_respond)
        case "round-robin":
            return syndisco.turn_manager.RoundRobin()
        case "random":
            return syndisco.turn_manager.RandomWeighted(p_respond=0)
        case _:
            raise ValueError(
                f"Unrecognizable turn manager type: {turn_manager_type}"
            )


def run_discussion_experiment(
    experiment: syndisco.experiments.DiscussionExperiment,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Starting synthetic discussion experiments...")
    experiment.begin(discussions_output_dir=output_dir, verbose=False)
    logger.info("Finished synthetic discussion experiments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations"
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--user-model-url",
        required=True,
        help=(
            "HuggingFace url for the desired participant (non-moderator) "
            "model. No GGUF support."
        ),
    )
    parser.add_argument(
        "--user-model-pseudo",
        required=True,
        help="Short-hand name for the participant (non-moderator) model",
    )
    parser.add_argument(
        "--mod-model-url",
        required=True,
        help=(
            "HuggingFace url for the desired moderator "
            "model. No GGUF support."
        ),
    )
    parser.add_argument(
        "--mod-model-pseudo",
        required=True,
        help="Short-hand name for the moderator model",
    )
    parser.add_argument(
        "--mod-strategy-file",
        required=True,
        help="A txt file containing the instructions for the moderator",
    )
    parser.add_argument("--user-instruction-path", required=True)
    parser.add_argument("--user-persona-path", required=True)
    parser.add_argument(
        "--turn-manager",
        required=True,
        choices=["random-weighted", "round-robin", "random"],
        help="The turn strategy used",
    )
    parser.add_argument(
        "--num-experiments", type=int, required=False, default=20
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="The turn strategy used",
    )
    parser.add_argument(
        "--trolls-active", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--mod-active", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    main(
        config_file_path=Path(args.config_file),
        user_model_url=args.user_model_url,
        user_model_name=args.user_model_pseudo,
        mod_model_url=args.mod_model_url,
        mod_model_name=args.mod_model_pseudo,
        mod_active=args.mod_active,
        turn_manager_type=args.turn_manager,
        mod_strategy_path=Path(args.mod_strategy_file),
        output_dir=Path(args.output_dir),
        num_experiments=args.num_experiments,
        trolls_active=args.trolls_active,
        user_instruction_path=Path(args.user_instruction_path),
        user_persona_path=Path(args.user_persona_path),
    )
