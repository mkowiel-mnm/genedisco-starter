# Source code of 2nd place solution in Maximize target discovery rate task (Task 1) GeneDisco ICLR-22 Challenge

The designed Acqusition Function selects a batch of `N` samples as a random selection from `4*N` samples chosen by the basic acquisition functions described in the benchmark:
`badge_acquisition`, `core_set_acquisition`, `kmeans_acquisition`, `top_uncertain_acquisition`.


![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-1.0.0-blue)

The starter repository for submissions to the [GeneDisco challenge](https://www.gsk.ai/genedisco-challenge/) for optimized experimental design in genetic perturbation experiments.

[GeneDisco (to be published at ICLR-22)](https://arxiv.org/abs/2110.11875) is a benchmark suite for evaluating active 
learning algorithms for experimental design in drug discovery. 
GeneDisco contains a curated set of multiple publicly available experimental data sets as well as open-source 
implementations of state-of-the-art active learning policies for experimental design and exploration.

## Install

```bash
pip install -r requirements.txt
```

## Use

### Setup

- Create a cache directory. This will hold any preprocessed and downloaded datasets for faster future invocation.
  - `$ mkdir /path/to/genedisco_cache`
  - _Replace the above with your desired cache directory location._
- Create an output directory. This will hold all program outputs and results.
  - `$ mkdir /path/to/genedisco_output`
  - _Replace the above with your desired output directory location._

### How to Run the Full Benchmark Suite?

Experiments (all baselines, acquisition functions, input and target datasets, multiple seeds) included in GeneDisco can be executed sequentially for e.g. acquired batch size `64`, `8` cycles and a `bayesian_mlp` model using:
```bash
run_experiments \
  --cache_directory=/path/to/genedisco_cache  \
  --output_directory=/path/to/genedisco_output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --max_num_jobs=1
```
Results are written to the folder at `/path/to/genedisco_cache`, and processed datasets will be cached at `/path/to/genedisco_cache` (please replace both with your desired paths) for faster startup in future invocations.


Note that due to the number of experiments being run by the above command, we recommend execution on a compute cluster.<br/>
The GeneDisco codebase also supports execution on slurm compute clusters (the `slurm` command must be available on the executing node) using the following command and using dependencies in a Python virtualenv available at `/path/to/your/virtualenv` (please replace with your own virtualenv path):
```bash
run_experiments \
  --cache_directory=/path/to/genedisco_cache  \
  --output_directory=/path/to/genedisco_output  \
  --acquisition_batch_size=64  \
  --num_active_learning_cycles=8  \
  --schedule_on_slurm \
  --schedule_children_on_slurm \
  --remote_execution_virtualenv_path=/path/to/your/virtualenv
```

Other scheduling systems are currently not supported by default.

### How to Run A Single Isolated Experiment (One Learning Cycle)?

To run one active learning loop cycle, for example, with the `"topuncertain"` acquisition function, the `"achilles"` feature set and
the `"schmidt_2021_ifng"` task, execute the following command:
```bash
active_learning_loop  \
    --cache_directory=/path/to/genedisco/genedisco_cache \
    --output_directory=/path/to/genedisco/genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="topuncertain" \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
```


### How to Evaluate a Custom Acquisition Function?

To run a custom acquisition function, set `--acquisition_function_name="custom"` and `--acquisition_function_path` to the file path that contains your custom acquisition function (e.g. [main.py](src/main.py) in this repo).
```bash
active_learning_loop  \
    --cache_directory=/path/to/genedisco/genedisco_cache \
    --output_directory=/path/to/genedisco/genedisco_output \
    --model_name="bayesian_mlp" \
    --acquisition_function_name="custom" \
    --acquisition_function_path=/path/to/src/main.py \
    --acquisition_batch_size=64 \
    --num_active_learning_cycles=8 \
    --feature_set_name="achilles" \
    --dataset_name="schmidt_2021_ifng" 
```

...where `"/path/to/custom_acquisition_function.py"` contains code for your custom acquisition function corresponding to the [BaseBatchAcquisitionFunction interface](genedisco/active_learning_methods/acquisition_functions/base_acquisition_function.py), e.g.:

```python
import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import \
    BaseBatchAcquisitionFunction

class RandomBatchAcquisitionFunction(BaseBatchAcquisitionFunction):
    def __call__(self,
                 dataset_x: AbstractDataSource,
                 batch_size: int,
                 available_indices: List[AnyStr], 
                 last_selected_indices: List[AnyStr] = None, 
                 model: AbstractBaseModel = None,
                 temperature: float = 0.9,
                 ) -> List:
        selected = np.random.choice(available_indices, size=batch_size, replace=False)
        return selected
```
Note that the last class implementing `BaseBatchAcquisitionFunction` is loaded by GeneDisco if there are multiple valid acquisition functions present in the loaded file.

## Submission instructions

For submission, you will need two things:

- A command line container build tool (Docker or Podman are the most common)
  - [Docker Installation Guide](https://docs.docker.com/get-docker/)
  - [Podman Installation Guide](https://podman.io/getting-started/installation)
- The eval.ai-CLI tool, which can be installed with pip
  - [EvalAI-CLI Documentation](https://cli.eval.ai/)
  - Run `$ pip install evalai`

Please note that all your submitted code must either be loaded via a dependency in `requirements.txt` or be present in the `src/` 
directory in this starter repository for the submission to succeed.

Once you have set up your submission environment, you will need to create a lightweight container image that contains your acquisition function.

### Submission steps

- Navigate to the directory to which you have cloned this repo to.
  - `$ cd /path/to/genedisco-starter`
- Ensure you have ONE acquisition function (inheriting from `BaseBatchAcquisitionFunction`) in [main.py](src/main.py)
  - _This is your pre-defined program entry point._ 
- Build your container image
  - `$ docker build -t submission:latest .`
- Save your image name to a shell variable
  - `$ IMAGE="submission:latest"`
- Use the EvalAI-CLI command to submit your image
  - Run the following command to submit your container image:
    - For Task 1: Maximize Target Discovery Rate
    - `$ evalai push $IMAGE --phase gsk-genedisco-test-1528`
    - For Task 2: Maximize Model Performance
    - `$ evalai push $IMAGE --phase gsk-genedisco-challenge-1528`
    - **Please note** that you have a maximum number of submissions that any submission will be counted against.
    - **Please note** that you will not receive your final private leaderboard score until the challenge period is over.

That’s it! Our pipeline will take your image and test your function.

If you have any questions or concerns, please reach out to us at [genedisco-challenge@gsk.com](mailto:genedisco-challenge@gsk.com?subject=[GeneDisco-Support-Request])


### Frequently Asked Questions (FAQ)

#### _"Will there be a public challenge leaderboard?"_

No. Participants are asked to compare their solutions internally against the provided baselines in the `genedisco` repository.
A final private leaderboard will be created using the submissions received via `eval.ai` after the challenge submission period is closed.

#### _"Which specific parts of the whole app lifecycle are we allowed to change?"_

You are able to change the acquisition function only (please see instructions above). We have chosen to fix the predictor and other aspects of the active learning loop in order to enable comparability of the developed solutions (there is a complex dependency between model, acquisition function and data).

#### _"How will submissions be scored?"_

We will score submissions against the two subtasks using the metrics calculated in `genedisco` - overall hit rate (% of top movers discovered) and model performance (mean squared model error) at the end of the active learning loop (after the last cycle).
Note that we will score the solutions against a held-out set of tasks not known to the participants via `genedisco`.

#### _"How do you define the best submission? Is it the last one or the best from all submitted?"_

We will use the last submission to calculate the team's score to avoid conferring an advantage to teams that produce more submissions.

#### _"Are the tasks evaluated independently? So I can Make 2 submissions with different Acquisition Functions and win in both subtasks?"_

Yes we have opened two subtasks in the challenge on [eval.ai](https://eval.ai/web/challenges/challenge-page/1528/submission) so that participating team's can submit independent solutions for each subchallenge (Task 1: Maximize Target Discovery Rate, Task 2: Maximize Model Performance).

#### _"The submissions limit (10) seems to be quite low. Especially that the evaluation measure in the challenge is not explicitly stated."_

The idea is for participants to develop and evaluate their own solutions internally against the many existing baselines already implemented in [GeneDisco](https://github.com/genedisco/genedisco) - hence there is no public leaderboard.
There will be a final private leaderboard that we will score through the eval.ai submission system.

#### _"Are the cycle number and the batch size fixed to 8 and 64 during the competition?"_

Yes, we will be fixing batch size and cycle number to numbers appropriate for the held-out tasks that the submissions will be evaluated against. You can expect them to be similar to the ones used in [the paper](https://arxiv.org/abs/2110.11875).

#### _"Will only the performances in the final cycle be considered for scoring?"_

Performance after the final iteration is the metric by which submissions will be compared. Interim steps will not be contributing to the final score.

#### _"For the competition, will we be provided the models' checkpoints or this is hidden from us?"_

The retrained model is exposed to your acquisition function at every iteration via the model parameter to the acquisition function method invocation. You are free to use it as you see fit and as the API allows. However, please note that you will not receive the checkpoint files from us as your acquisition function will be running end-to-end on our cloud servers for evaluation (there is no interactive mode for participants; the code is considered final once submitted).

## Citation

Please consider citing, if you reference or use our methodology, code or results in your work:

    @inproceedings{mehrjou2022genedisco,
        title={{GeneDisco: A Benchmark for Experimental Design in Drug Discovery}},
        author={Mehrjou, Arash and Soleymani, Ashkan and Jesson, Andrew and Notin, Pascal and Gal, Yarin and Bauer, Stefan and Schwab, Patrick},
        booktitle={{International Conference on Learning Representations (ICLR)}},
        year={2022}
    }

### License

[License](LICENSE.txt)

### Authors

Arash Mehrjou, GlaxoSmithKline plc<br/>
Jacob A. Sackett-Sanders, GlaxoSmithKline plc<br/>
Patrick Schwab, GlaxoSmithKline plc<br/>

Marcin Kowiel, MNM Bioscience <br/>

### Acknowledgements

PS, JSS and AM are employees and shareholders of GlaxoSmithKline plc.
