import uuid
import os
import argparse

from sdl.serialization import annotation_io, persona
from sdl.util.file_util import read_file


def generate_annotator_file(
    annotator_persona: persona.LlmPersona,
    instructions: str,
    history_ctx_len: int,
    include_moderator_comments: bool,
) -> annotation_io.LlmAnnotationData:
    """Generate an annotation configuration object from provided attributes.
    The object can then be used for IO operations or directly as input for a conversation.

    :param annotator_personas: a list of all personas in JSON/dict format, from which a random subset will be selected depending on num_users
    :type annotator_personas: list[LlmPersona]

    :return: An IO conversation configuration object which can be used for persistence, or as input for a conversation
    :rtype: conversation_io.LLMConvData
    """
    data = annotation_io.LlmAnnotationData(
        attributes=annotator_persona.to_attribute_list(),
        instructions=instructions,
        history_ctx_len=history_ctx_len,
        include_moderator_comments=include_moderator_comments,
    )
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate conversation configs using modular configuration files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for generated annotation config files",
    )
    parser.add_argument(
        "--persona_dir",
        required=True,
        help="Directory containing JSON files for LLM annotator personas",
    )
    parser.add_argument(
        "--instruction_path",
        required=True,
        help="Path to .txt file containing annotator instructions",
    )
    parser.add_argument(
        "--history_ctx_len",
        type=int,
        default=4,
        help="How many previous comments the annotator will remember.",
    )
    parser.add_argument(
        "--include_mod_comments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to include moderator comments in the annotations and the conversational context. "
        "Setting this flag to false will also impact user annotations, since it changes which previous "
        "comments the annotator can see.",
    )
    args = parser.parse_args()

    print("Reading input files...")
    persona_files = os.listdir(args.persona_dir)
    personas = [
        persona.LlmPersona.from_json_file(os.path.join(args.persona_dir, persona_file))
        for persona_file in persona_files
    ]
    instructions = read_file(args.instruction_path)

    print("Processing...")
    for llm_persona in personas:
        annotation_config_file = generate_annotator_file(
            annotator_persona=llm_persona,
            instructions=instructions,
            history_ctx_len=args.history_ctx_len,
            include_moderator_comments=args.include_mod_comments
        )
        annotation_config_file.to_json_file(
            os.path.join(args.output_dir, str(uuid.uuid4()) + ".json")
        )
    print("Files exported to " + args.output_dir)


if __name__ == "__main__":
    main()
