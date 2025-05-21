# The Virtual Moderation Dataset

Presented in ["Scalable Evaluation of Online Facilitation Strategies via Synthetic Simulation of Discussions"](https://arxiv.org/abs/2503.16505).

This dataset contains **synthetically generated** discussions and annotations using exclusively Large Language Model (LLM) agents. Discussions are performed between randomly selected users, with a LLM moderator/facilitator following various facilitation strategies.

Each LLM user and annotator use a distinct *Sociodemographic Background*. User-agents also have different *roles* when joining the discussion. Each discussion consists of 28 comments - 14 participant comments and 14 moderator interventions. Each comment is annotated by 10 different LLM annotators for *toxicity* and *argument quality*. 

Designed to analyze facilitation strategies and synthetic online discussion simulation but can also be used for finetuning LLM facilitators.