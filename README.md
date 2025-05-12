# Evaluating Online Moderation Strategies Through Synthetic Discussion Generation

Synthetic dataset generation using the [SynDisco](https://github.com/dimits-ts/synthetic_discussion_framework) library. Experiments exploring the effect of various LLM moderation strategies in online conversations.


## Project Structure

* [`run.py`](run.py) Execution script
* [`data/`](data/) Input and output data of the experiments
  * [`data/annotation_input/`](data/annotation_output/) Configuration files for the synthetic annotation jobs
  * [`data/annotation_output/`](data/annotation_output/) Annotation output (JSON format)
  * [`data/discussions_input/`](data/discussions_input/) Configuration files for the synthetic discussion jobs
  * [`data/discussions_output/`](`data/discussions_output/`) Logs of the executed synthetic discussions (JSON format)
  * [`data/datasets/`](data/datasets) Exported CSV datasets (original + ablation)
  * [`data/run_configs/`](data/run_configs/) YAML configuration files for running the experiments

* [`notebooks/`](`notebooks/`) Analyzing the experiments
  * [`notebooks/tasks/`](notebooks/tasks/) Shared notebook modules
  * [`notebooks/ablation.ipynb`](notebooks/ablation.ipynb) Ablation study
  * [`notebooks/moderation.ipynb`](notebooks/moderation.ipynb) Moderator intervention analysis
  * [`notebooks/timeseries.ipynb`](notebooks/timeseries.ipynb) Timeseries analysis of synthetic discussions
  * [`notebooks/toxicity_aq.ipynb`](notebooks/toxicity_aq.ipynb) General discussion quality analysis
  
* [`graphs/`](graphs/) Exported graphs used in the paper
