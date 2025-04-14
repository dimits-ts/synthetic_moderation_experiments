from pathlib import Path


GRAPH_OUTPUT_DIR = Path("../graphs")

MODERATION_STRATEGY_MAP = {
    "vanilla": "No Instructions",
    "moderation_game": "Moderation Game",
    "no_moderator": "No Moderator",
    "erulemaking": "Human Mod. Guidelines",
    "constructive_communications": "Human Fac. Guidelines",
    "collective_constitution": "Rules Only",
}

MODEL_MAP = {
    "hardcoded": "hardcoded",
    "mistral-nemo-abliterated": "Mistral Nemo (abl.)",
    "qwen2.5-32b-4bit": "Qwen 2.5",
    "llama-3.1-70b-4bit": "LLaMa 3.1",
}

METRIC_MAP = {
    "toxicity": "Toxicity",
    "argumentquality": "Argument Quality"
}
METRICS = METRIC_MAP.values()
