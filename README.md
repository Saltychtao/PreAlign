This is the repo for the paper "PreAlign Boosting Cross-Lingual Transfer by Early Establishment of Multilingual Alignment"

# Requirements
Our code is built on huggingface transformers.

- python>=3.8.0
- pytorch>=2.1.0
- transformers >= 4.34.0

# PreAlign Pipeline

## Step 1: Perform multilingual alignment based on collected multilingual dictionaries
You should first prepare multilingual dictionaries, where each line of the file is in the form of `SRCWORD\tTGTWORD`. You need also prepare a language modeling dataset in the huggingface datasets format, with each entry being `{"text": "pretraining text here..."}`.

After that, we can perform the multilingual alignment using the following command:

```
bash shell_scripts/multilingual_alignment.sh \
--lm_train_file YOUR_LM_DATASET \
--dict_file YOUR_DICT_FILE \
--output_dir YOUR_OUTPUT_DIR \
--hf_config YOUR_MODEL_CONFIG \
--config_file configs/base/CONFIG_FILE \
--tokenizer_path YOUR_MODEL_TOKENIZER \
--devices 0,1,2,3,4,5,6,7
```

We have prepare several config file in the directory `configs/base/`.

## Step 2: Perform language modeling task with codeswitch
After obtaining the model with multilingual aligned, we can then train the model to perform language modeling task:

```
bash shell_scripts/pretrain_with_codeswitch.sh \
--lm_train_file YOUR_LM_DATASET \
--dict_file YOUR_DICT_FILE \
--output_dir YOUR_OUTPUT_DIR \
--hf_config YOUR_MODEL_CONFIG \
--config_file configs/base/CONFIG_FILE \
--tokenizer_path YOUR_MODEL_TOKENIZER \
--devices 0,1,2,3,4,5,6,7 \
--codeswitch_ratio 0.05 
```
# Citation
If you find this repo useful, please feel free to leave a star and cite our paper:
```
@misc{li2024prealign,
      title={PreAlign Boosting Cross-Lingual Transfer by Early Establishment of Multilingual Alignment}, 
      author={Jiahuan Li and Shujian Huang and Xinyu Dai and Jiajun Chen},
      year={2024},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```




