# Evaluating Online Moderation Strategies Through Synthetic Discussion Generation

Synthetic dataset generation using the [SynDisco](https://github.com/dimits-ts/synthetic_discussion_framework) library. Experiments exploring the effect of various LLM moderation strategies in online conversations.


## Usage

To download the code, run `git clone --recurse-submodules -j8 https://github.com/dimits-ts/synthetic_moderation_experiments.git`


### Running synthetic experiments

To generate your own experiments first configure your master configuration file ([example](data/server_config.yml)).

Then point the fields on the master configuration file, to this project's configurations (see **Project Structure**) or create your own! ([example discussion configurations](data/discussions_input/), [example annotation configurations](data/annotation_input/))

Lastly, execute:
```bash
cd syntetic_discussions_framework/src
python run.py --config_file "../../data/server_config.yml"
```

For more information, visit [SynDisco's documentation](https://github.com/dimits-ts/synthetic_discussion_framework/blob/master/README.md).

### Running analysis on the exported dataset

To run the code for analysis and graph generation, run [`notebooks/toxicity_analysis.ipynb`](notebooks/toxicity_analysis.ipynb)


### Environment

TODO


## Project Structure

* [`data/`](data/) input and output data of the experiments
  * [`data/dataset.csv`](data/dataset.csv) the synthetic dataset (includes discussions and annotations)
  * [`data/server_config.yml`](data/server_config.yml) the master configuration file
  * [`data/annotation_input/`](data/annotation_output/) configuration files for the synthetic annotation jobs
  * [`data/annotation_output/`](data/annotation_output/) annotation output (JSON format)
  * [`data/discussions_input/`](data/discussions_input/) configuration files for the synthetic discussion jobs
  * [`data/discussions_output/`](`data/discussions_output/`) logs of the executed synthetic discussions (JSON format)

* [`notebooks/`](`notebooks/`) contains the notebook responsible for analyzing the experiments
  * [`notebooks/toxicity_analysis.ipynb`](notebooks/toxicity_analysis.ipynb) dataset analysis
  * [`notebooks/tasks/`](notebooks/tasks/) contains notebook-specific code

* `synthetic_discussion_framework/` the [SynDisco](https://github.com/dimits-ts/synthetic_discussion_framework) framework

* [`graphs/`](graphs/) exported graphs from [`notebooks/toxicity_analysis.ipynb`](notebooks/toxicity_analysis.ipynb)
