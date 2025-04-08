# Experiment Environment for Evaluating ReadAgent

This repository is an experiment environment to evaluate ([ReadAgent](https://read-agent.github.io/)) by Lee et al. (2024) as done in our paper:

**Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models** (Laitenberger et al., 2025) - [Paper (TODO: add link)](arxiv.org) - [Repo-Overview](https://github.com/Lightnz/stronger-baselines-rag/)

We have used the implementation provided in the notebook on the official [ReadAgent Github Page](https://github.com/read-agent/read-agent.github.io/blob/main/assets/read_agent_demo.ipynb). 

We made a few adjustments for optimzation and comparability, including:

- using nltk split sentences as the smallest unit for creating "pages", since we observed, that datasets do not safely split paragraphs with "\n"


## Setup

### Requirements

Before using this repository, ensure Python 3.11+ is installed. You can install the requirements with:

```bash
pip install -r requirements.txt
```

### OpenAI API Key

We used the OpenAI API for our experiments. For reproducing the experiments you need to create a file named `config.py` in the project root directory (next to `requirements.txt`) and add the line `OPENAI_API_KEY = "your-api-key"` with your API key. The file is excluded from repo updates in the `.gitignore`, so don't worry about exposing the key in case you plan to commit.

## Bash commands & Logs

All provided bash commands are to be executed from the project's root directory.
Provided scripts generally create log-files under `experiments/logs/`.

## Experiments

### Datasets

QuALITY: Question Answering with Long Input Texts, Yes! (Pang et al., 2022) - [Paper](https://arxiv.org/pdf/2112.08608) - [Github](https://github.com/nyu-mll/quality?tab=readme-ov-file)

∞bench: Extending long context evaluation beyond 100K tokens. (Zhang et al., 2024) - [Paper](https://arxiv.org/abs/2402.13718) - [Github](https://github.com/OpenBMB/InfiniteBench)

The NarrativeQA reading comprehension challenge (Kočiský et al., 2018) - [Paper](https://arxiv.org/abs/1712.07040) - [Github](https://github.com/google-deepmind/narrativeqa)


### Preparing datasets

- Prepare data folders  
    ```bash
    mkdir -p data/{quality,narrativeqa,infinity_bench}/raw
    ```

#### QuALITY (development set)
- Download and unzip [QuALITY v1.0.1](https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip) 

- Copy the file `QuALITY.v1.0.1.htmlstripped.dev` into `/data/quality/raw`

- Run the preprocessing script to group the dataset by documents.
    ```bash
    python -m source.data.quality.preprocess_quality
    ```
    It should have created a new json file under the `data/quality/preprocessed` path.

#### ∞bench (En.MC)
- Download the [longbook_choice_eng.jsonl](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_choice_eng.jsonl) file directly from huggingface into the `data/infinity_bench/raw` folder.

- Run the preprocessing script:
    ```bash
    python -m source.data.infinity_bench.preprocess_longbook_choice_eng
    ```
    It should have created a new json file under `data/infinity_bench/preprocessed`.

#### NarrativeQA
- Clone the repository, e.g. to your home directory
    ```bash
    cd ~
    git clone git@github.com:google-deepmind/narrativeqa.git
    ```

- Execute the `download_stories.sh` script in the NarrativeQA repository to download the 1572 documents into the `tmp` directory. This might take a while.
    ```bash
    cd ~/narrativeqa #or your clone destination

    bash download_stories.sh
    ```

- Back in the raptor-eval repository run the preprocessing script to prepare the QA experiment.
You might need to change the `NARRATIVE_QA_PATH` variable in case you did not clone the NarrativeQA repository into your home directory. 
    ```bash
    python -m source.data.narrative_qa.preprocess_narrative
    ```
    It should have created the file `processed_qaps_test.json` in the `data/narrativeqa/preprocessed` folder.

### 📃 Precreate pages

#### 💰 Budget

The budget requirements outlined below are approximate estimates and should be considered as rough guidelines. If you plan to reproduce the experiments, please conduct your own calculations to ensure the costs align with your budget.

| Dataset     | Budget Level     | Estimated input tokens              |
|-------------|------------------|---------------------------------------|
| QuALITY     |  ~ $0.15         | 115 docs x 5.1k avg tokens x 2 = 1M       |
| ∞bench      |  ~ $4 - $8       | 58 docs x 184k avg tokens x 2 = 22M       |
| NarrativeQA |  ~ $8 - $16       | 355 docs x 57.7k avg tokens x 2 = 40M     |


#### 📝 Scripts
We prepared the scripts `precreate_pages.py` for each dataset to precreate all ReadAgent pages and shortened_pages for the experiments. It uses GPT-4o-mini by default.

Upon completion there should be created pages and shortened_pages in the output folders `experiments/artifacts/pages/<dataset>/...` and `experiments/artifacts/shortened_pages/<dataset>/...`

Run the scripts with:

##### QuALITY
```bash
python -m source.experiments.quality.precreate_pages
```

##### ∞bench
```bash
python -m source.experiments.infinity_bench.longbook_choice_eng.precreate_pages
```

##### NarrativeQA
You might need to change `NARRATIVEQA_PATH` in the script and adapt it to the location of your cloned NarrativeQA repository with the downloaded documents in the `tmp` folder.
```bash
python -m source.experiments.narrative_qa.precreate_pages
```

### 🚀 Run Experiments

#### 💰 Budget

The budget requirements outlined below are approximate estimates and should be considered as rough guidelines. If you plan to reproduce the experiments, please conduct your own calculations to ensure the costs align with your budget.

To calculate a sample necessary input token amount we took the estimate of: number of questions x measured average input tokens/question.

The following table concludes the budget level estimates for all datasets.


| Dataset     | Budget Level (GPT4o-mini)    | Budget Level (GPT4o) | Approximate Input tokens              |
|-------------|------------------|------------------|---------------------------------------|
| QuALITY     |  $ 2           | $ 25              | 2k questions x 5k average tokens/question = 10M        |
| ∞bench      |  $ 3           | $ 50              | 225 questions x 87k average tokens/question = 20M       |
| NarrativeQA |  $54 | $893 (not executed) | 10.5k questions 34k average tokens/question = 357M |


#### 📝 Scripts

We provide scripts called `run_experiment.py` under `experiments_source/<dataset>/` to run the experiments on the previosuly created pages.

Before executing you need to make a few adjustments in each script:

- the scripts use GPT-4o-mini by default. To conduct experiments with GPT-4o, adjust `OPENAI_MODELSTRING`. The model strings for GPT-4o-mini and GPT-4o are provided as comments and can be easily switched by commenting/uncommenting the desired option.

- change `STORED_PAGES_FOLDER_PATH` and `STORED_SHORTENED_PAGES_FOLDER_PATH` to match your precreated pages folders.

- the scripts use parallelity to run the experiments on multiple documents at the same time. If you want to run the experiment sequentially, or control the amount of parallelity find the line `with ThreadPoolExecutor() as executor:` and adjust it accordingly (e.g., `ThreadPoolExecutor(max_workers=1)` to run sequentially).

- you can set the hyperparameters for the experiment by modifying the experiments list in the run_experiment_batch() function. `max_pages = 6` defines the maximum of pages the model is allowed to look up. We used the setting that was used in the official ReadAgent repository, which was reported as the best performing.

The scripts output the model's answers into a `jsonl` file under `experiments/artifacts/answers/<dataset>`. 

For full reproducibility, we provide our experiment hyperparameter settings in the script.

Run the scripts with:

##### QuALITY
```bash
python -m source.experiments.quality.run_experiment
```

##### ∞bench
```bash
python -m source.experiments.infinity_bench.longbook_choice_eng.run_experiment
```

##### NarrativeQA
```bash
python -m source.experiments.narrative_qa.run_experiment
```

### 📊 Evaluate

We prepared scripts to evaluate stored answer files conveniently.
You can use them with multiple answer files in the `experiments/artifacts/answers/<dataset>` folder.

Run:

#### QuALITY
```bash
python -m source.experiments.quality.eval
```
#### ∞bench
```bash
python -m source.experiments.infinity_bench.longbook_choice_eng.eval
```

#### NarrativeQA
```bash
python -m source.experiments.narrative_qa.eval
```

#### Results
The scripts create a `json` and `csv` file in the folder of the respective answer files.

E.g., the json-file for QuALITY results looks like this:
```bash
    "file": "2025-04-07_15-02-read-agent-quality-dev_0_m-lu-pages-6_gpt-4o-mini-2024-07-18.jsonl",
    "total_entries": 2086,
    "correct": 1651,
    "accuracy": 79.15,
    "avg_tokens": 4775.36,
    "hard_entries": 1065,
    "hard_correct": 748,
    "hard_accuracy": 70.23,
    "non_hard_entries": 1021,
    "non_hard_correct": 903,
    "non_hard_accuracy": 88.44
```

`avg_tokens`: Since we track the exact amount of input tokens for each evaluated question we can calculate the average input tokens per question, which is a useful metric for the required budget and resulting efficiency.


## References
Alex Laitenberger, Christopher D. Manning and Nelson F. Liu. 2025. Stronger Baselines for Retrieval-Augmented Generation with Long-Context Language Models [Paper (TODO: add link)](www.arxiv.org) - [Github](https://github.com/Lightnz/stronger-baselines-rag/)

Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. 2024. A human-inspired reading agent with gist memory of very long contexts. In Proc. of ICML. - [Paper](https://arxiv.org/abs/2402.09727) - [Github](https://read-agent.github.io/)

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel R. Bowman. 2022. QuALITY: Question answering with long input texts, yes! In Proc. of NAACL. - [Paper](https://arxiv.org/abs/2112.08608) - [Github](https://github.com/nyu-mll/quality)

Tomáš Kočiský, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The NarrativeQA reading comprehension challenge. Transactions of the Asso-361
ciation for Computational Linguistics, 6:317–328. - [Paper](https://arxiv.org/abs/1712.07040) - [Github](https://github.com/google-deepmind/narrativeqa)

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024. ∞bench: Extending long context evaluation beyond 100K tokens. In Proc. of ACL. - [Paper](https://arxiv.org/abs/2402.13718) - [Github](https://github.com/OpenBMB/InfiniteBench)
