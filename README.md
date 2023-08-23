

# Eliciting Translation Ability for LLMs using Multilingual Finetuning with Translation Instructions

This is the repo for the paper "Eliciting the Translation Ability of Large Language Models via Multilingual Finetuning with Translation Instructions", which aims to investigate the process of eliciting translation ability of LLMs using instruction tuing, and studies how this ability generalizes to unseen language pairs.

## Requirements
This repo requires `transformers`, `datasets`, `deepspeed` and `langcodes` as the dependency. You can install them by running the following scripts:
```
bash install.sh
```

## Data Preparation

### Training Data
You should first download the [Wikimatrix](https://opus.nlpl.eu/WikiMatrix.php) corpus, and put into the folder `./dataset`, and prepare the translation-instruction-following datasets using the following command:
```
python3 dataset_scripts/prepare_dataset.py --infile /path/to/data --outfile /path/to/outfile --srclang $srclang -tgtlang $tgtlang
```

### Evaluation Data
You also need to download the [Flores101](https://huggingface.co/datasets/gsarti/flores_101) to evaluate the finetuned model.

## Training and Inferencing

Below is a command that fine-tunes XGLM-7B with dataset on a machine with 8 A100 80G GPUs in Deepspeed Stage-2 model.  It also report the translation performance measured by sacrebleu on all language pairs. `<your_data_path>` contains the instruction-following data preprocess in the previous steps. `<your_savedir>` specifies where the finetuned checkpoint should be saved. `<flores101_dataset>` contains the downloaded Flores-101 dev and test split.

```bash
bash shell_scripts/run.sh \
    --plm XGLM-7.5B \
    --train_file <your_data_path> 、
    --batch_size 10 --max_length 256 、
    --update_freq 1 \
    --savedir <your_savedir> \
    --devices 0,1,2,3,4,5,6,7 \
    --master_port 8888 \
    --save_steps 250 \
    --max_steps 2000 \
    --flores101_dataset <flores101_dataset>
```

Note the given training script is meant to be simple and easy to use, and is not particularly optimized.
To run on less or less capable gpus, e.g. V100, you may prefer to tune the `batch_size` and `gradient_accumulation_steps` to keep a global batch size of 128. Global batch size has not been tested for optimality.



### Citation

Please kindly cite our paper if you use the data or code in this repo.

```
@misc{li2023eliciting,
      title={Eliciting the Translation Ability of Large Language Models via Multilingual Finetuning with Translation Instructions}, 
      author={Jiahuan Li and Hao Zhou and Shujian Huang and Shanbo Cheng and Jiajun Chen},
      year={2023},
      eprint={2305.15083},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```