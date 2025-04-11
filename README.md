# Environment Setup

```bash
uv init

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv add polars
```

# 下载数据

因为 [NSI](https://www.hec.usace.army.mil/confluence/nsi/technicalreferences/latest/technical-documentation) 的数据是 2022 年的，所以
[ACS](https://www2.census.gov/programs-surveys/acs/data/pums/2022/1-Year/) 的数据也使用 2022 年的。

# Using your own dataset

## 准备数据

First, create a directory for your dataset in [./data](./data):

```bash
cd data
mkdir <NAME_OF_YOUR_DATASET>
```

Compile your raw tabular data in .csv format. **The first row should be the header** indicating the name of each column, and the remaining rows are records. After finishing these steps, place you data's csv file in the directory you just created and name it as <NAME_OF_YOUR_DATASET>.csv. 

Then, create <NAME_OF_YOUR_DATASET>.json in [./data/Info](./data/Info). Write this file with the metadata of your dataset, covering the following information:

```json
{
    "name": "<NAME_OF_YOUR_DATASET>",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/<NAME_OF_YOUR_DATASET>/<NAME_OF_YOUR_DATASET>.csv"
    "test_path": null,
}
```

Finally, run the following command to process your dataset:

```bash
# python process_dataset.py --dataname <NAME_OF_DATASET>
python process_dataset.py --dataname adult
```

对于我们的 acs 数据集，

```bash
python process_dataset_acs.py --dataname de # de, nc
```

## 训练 TabDiff


To train an unconditional TabDiff model across the entire table, run

```bash
# python main.py --dataname <NAME_OF_DATASET> --mode train
python main.py --dataname de_prepared --mode train
```

Current Options of ```<NAME_OF_DATASET>``` are: adult, default, shoppers, magic, beijing, news

Wanb logging is enabled by default. To disable it and log locally, add the ```--no_wandb``` flag.

To disable the learnable noise schedules, add the ```--non_learnable_schedule```. Please note that in order for the code to test/sample from such model properly, you need to add this flag for all commands below.

To specify your own experiment name, which will be used for logging and saving files, add ```--exp_name <your experiment name>```. This flag overwrites the default experiment name (learnable_schedule/non_learnable_schedule), so, similar to ```--non_learnable_schedule```, once added to training, you need to add it to all following commands as well.

## Sampling and Evaluating TabDiff (Density, MLE, C2ST)

To sample synthetic tables from trained TabDiff models and evaluate them, run

```bash
# python main.py --dataname <NAME_OF_DATASET> --mode test --report --no_wandb
python main.py --dataname de_prepared --mode test --report --no_wandb
```

This will sample 20 synthetic tables randomly. Meanwhile, it will evaluate the density, mle, and c2st scores for each sample and report their average and standard deviation. The results will be printed out in the terminal, and the samples and detailed evaluation results will be placed in ./eval/report_runs/<EXP_NAME>/<NAME_OF_DATASET>/.