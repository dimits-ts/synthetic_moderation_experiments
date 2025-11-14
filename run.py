import argparse
import logging
import sys
import yaml
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

import syndisco.model
import syndisco.actors
import syndisco.experiments
import syndisco.logging_util
import syndisco.turn_manager
import syndisco.postprocessing

logger = logging.getLogger(Path(__file__).name)


def main(config_file_path: Path):
    with open(config_file_path, "r", encoding="utf8") as file:
        yaml_data = yaml.safe_load(file)

    setup_logging(logging_config=yaml_data["logging"])

    model = syndisco.model.TransformersModel(
        model_path=yaml_data["discussion_model"]["model_path"],
        name=yaml_data["discussion_model"]["model_pseudoname"],
        remove_string_list=yaml_data["discussion_model"]["disallowed_strings"],
        max_out_tokens=yaml_data["discussion_model"]["max_tokens"],
    )
    discussion_exp = create_discussion_experiment(
        llm=model,
        discussion_config=yaml_data["discussions"],
    )
    run_discussion_experiment(
        experiment=discussion_exp,
        output_dir=Path(yaml_data["discussions"]["files"]["output_dir"]),
    )

    export_dataset(
        json_output_dir=Path(yaml_data["discussions"]["files"]["output_dir"]),
        dataset_export_dir=Path(
            yaml_data["discussions"]["files"]["export_dir"]
        ),
        dataset_name="test.csv",
    )


def export_dataset(
    json_output_dir: Path, dataset_export_dir: Path, dataset_name: str
) -> None:
    dataset_export_dir.mkdir(exist_ok=True, parents=True)
    export_path = dataset_export_dir / dataset_name
    conv_df = syndisco.postprocessing.import_discussions(json_output_dir)
    conv_df.to_csv(path_or_buf=export_path, encoding="utf8", mode="w+")
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
    llm, discussion_config: dict
) -> syndisco.experiments.DiscussionExperiment:
    context = discussion_config["experiment_variables"]["context_prompt"]

    topics = get_topics(
        topics_dir=Path(discussion_config["files"]["topics_dir"])
    )

    users = get_users(
        model=llm,
        context=context,
        instructions_path=Path(
            discussion_config["files"]["user_instructions_path"]
        ),
        html_dir=Path(discussion_config["files"]["user_persona_dir"]),
    )

    include_mod = discussion_config["experiment_variables"]["include_mod"]
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
            model=llm,
            mod_persona=mod_persona,
            context=context,
            mod_instructions_path=Path(
                discussion_config["files"]["mod_instructions_path"]
            ),
        )

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


def get_topics(topics_dir: Path) -> list[str]:
    topics = []
    for file in topics_dir.iterdir():
        topic = file.read_text()
        topics.append(topic)
    return topics


def parse_profile_table(html: str) -> list[syndisco.actors.Persona]:
    """Parse the given HTML table into a list of Persona objects."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tbody tr")

    personas = []

    for row in rows:
        cols = row.select("td")
        if len(cols) < 4:
            continue

        # --- ID ---
        pid = cols[0].get_text(strip=True)

        # --- Demographics ---
        demo_divs = cols[1].select("div.text-sm > div")
        age, sex = None, None
        demographic_group = None
        if len(demo_divs) >= 3:
            # Example: "25-34 Male" or "Under 18 Female"
            age_sex = demo_divs[0].get_text(strip=True).split()
            if len(age_sex) >= 2:
                age_range = age_sex[0]
                if age_range.isdigit():
                    age = int(age_range)
                else:
                    # Use the midpoint if age is a range like "25-34"
                    import re

                    m = re.match(r"(\d+)-(\d+)", age_range)
                    if m:
                        age = (int(m.group(1)) + int(m.group(2))) // 2
                sex = age_sex[-1]
            demographic_group = demo_divs[1].get_text(strip=True)
            sexual_orientation = demo_divs[2].get_text(strip=True)
        else:
            sexual_orientation = ""

        # --- Background ---
        bg_divs = cols[2].select("div.text-sm > div")
        if len(bg_divs) >= 3:
            education_level = bg_divs[0].get_text(strip=True)
            current_employment = bg_divs[1].get_text(strip=True)
        else:
            education_level = current_employment = ""

        # --- Personality ---
        personality_desc = cols[3].select_one(".text-foreground\\/90")
        description = (
            personality_desc.get_text(strip=True) if personality_desc else ""
        )
        tags = [t.get_text(strip=True) for t in cols[3].select("span.rounded")]

        # Create Persona object
        persona = syndisco.actors.Persona(
            username=pid,
            age=age if age is not None else -1,
            sex=sex or "",
            sexual_orientation=sexual_orientation,
            demographic_group=demographic_group or "",
            current_employment=current_employment,
            education_level=education_level,
            special_instructions=description,
            personality_characteristics=tags,
        )

        personas.append(persona)

    return personas


def get_personas_from_html(html_dir: Path) -> list[syndisco.actors.Persona]:
    personas = []
    for file in html_dir.iterdir():
        html = file.read_text()
        personas += parse_profile_table(html)
    return personas


def get_users(
    model: syndisco.model.BaseModel,
    context: str,
    instructions_path: Path,
    html_dir: Path,
) -> list[syndisco.actors.Actor]:
    personas = get_personas_from_html(html_dir=html_dir)
    instructions = instructions_path.read_text()

    actors = []
    for persona in personas:
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
    if turn_manager_type == "random_weighted":
        return syndisco.turn_manager.RandomWeighted(p_respond=p_respond)
    else:
        return syndisco.turn_manager.RoundRobin()


def run_discussion_experiment(
    experiment: syndisco.experiments.DiscussionExperiment, output_dir: Path
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Starting synthetic discussion experiments...")
    experiment.begin(discussions_output_dir=output_dir)
    logger.info("Finished synthetic discussion experiments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic conversations"
    )
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    main(config_file_path=Path(args.config_file))
