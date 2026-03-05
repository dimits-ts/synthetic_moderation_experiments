
# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

from pathlib import Path


GRAPH_OUTPUT_DIR = Path("../graphs")
DATASET_DIR = Path("../data/datasets")
HATCHES = ["/", "|", "\\", "-", "x", "o", "+", "O", ".", "*"]

MODERATION_STRATEGY_MAP = {
    "vanilla": "No Instructions",
    "moderation_game": "Mod. Game",
    "no_moderator": "No Moderator",
    "erulemaking": "Reg. Room",
    "constructive_communications": "Constr. Comms",
    "collective_constitution": "Rules Only",
}

MODEL_MAP = {
    "hardcoded": "hardcoded",
    "mistral-nemo-4bit": "Mistral Nemo",
    "qwen2.5-32b-4bit": "Qwen 2.5",
    "llama-3.1-70b-4bit": "LLaMa 3.1",
}

METRIC_MAP = {
    "toxicity": "Toxicity",
    "argumentquality": "Argument Quality"
}
METRICS = METRIC_MAP.values()
