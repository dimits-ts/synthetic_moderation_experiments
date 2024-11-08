# Synthetic Moderation Experiments

Synthetic dataset generation using the [Synthetic Discussion Framework (SDF)](https://github.com/dimits-ts/synthetic_discussion_framework). Experiments exploring the effect of various LLM moderation strategies in online conversations.

## Usage

To download the code, run `git clone --recurse-submodules -j8 https://github.com/dimits-ts/synthetic_moderation_experiments.git`

Usage instructions can be found in the [SDF's documentation](https://github.com/dimits-ts/synthetic_discussion_framework/blob/master/README.md).

### Environment

We use the environment outlined in the [SDF's documentation](https://github.com/dimits-ts/synthetic_discussion_framework/blob/master/README.md)

### Setting up the LLM

Run [`src/scripts/download_model.sh`](src/scripts/download_model.sh) in order to download the model used to run the framework in the correct directory (~5 GB of storage needed). By default, the model used is the `llama-3.8b-instruct.gguf`, but any model supported by the `llama_cpp` library may be used. A complete list of models can be found [here](https://github.com/ggerganov/llama.cpp).

## Project Structure

* `data/`
  * `data/annotation_input/` contains the configuration files for the annotation jobs
  * `data/generated_discussions_input/` contains the configuration files for the synthetic conversation experiments
  * `data/generated_discussions_output/` contains the logs (dataset) of the executed synthetic conversations
  * `data/annotation_output/` contains the logs (dataset) of the executed synthetic annotation of the synthetic conversations

* `scripts/`
  * `scripts/download_model.sh` downloads the specified model (see above) in the standard directory
  * `scripts/*_personalized.sh` experiment-specific configurations of the analogous SDF scripts

* `models/` - auto-generated directory containing the LLMs
* `logs/` - auto-generated directory containing execution logs
