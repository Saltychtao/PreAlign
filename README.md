This is the repo for the paper "PreAlign Boosting Cross-Lingual Transfer by Early Establishment of Multilingual Alignment"

# Requirements
Our code is built on huggingface transformers.

- python>=3.8.0
- pytorch>=2.1.0
- transformers == 4.31.0
- tokenizers == 0.12.0

# PreAlign Pipeline

## Step 1: Perform multilingual alignment based on collected multilingual dictionaries
You should first prepare language dictionaries. For performing one-to-one mapping, the format should be like `{"src_text": apple, "tgt_text": 苹果}` at each line. For aligning to multiple translations/languages simutanously, the format should be like `{"words": "bank $ 银行 $ banque", "index": 1}`. See `data/en-zh-example.multi.json` for reference.

After that, we can perform the one-to-one alignment using the following command:

```
bash shell_scripts/multilingual_alignment.sh \
--lm_train_file YOUR_LM_DATASET \
--dict_train_file YOUR_DICT_FILE \
--output_dir YOUR_OUTPUT_DIR \
--model_name_or_path MODEL_NAME_OR_PATH
--tokenizer_path YOUR_MODEL_TOKENIZER \
--devices 0,1,2,3,4,5,6,7
```

For multilingual alignment, you should add additional args `--contrastive_multi 1`.

Note that, you should decide how the space token is represented in your tokenizer. For example, Llama represents spaces using "▁", while pythia uses "Ġ". Find the token, and add them at the Line 33/41 in `src/data/lm_data.py`. For multilingual alignment, you should also decide how the `$` symbol is mapped, and add the corresponding indices in the function `tokenize_parallel_function` in `src/data/bilingual_dictionary_pretrain_data_multi.py`.

## Step 2: Perform language modeling task with codeswitch
After obtaining the model with multilingual aligned, we can then train the model to perform language modeling task:

```
bash shell_scripts/pretrain.sh \
--train_file YOUR_LM_DATASET \
--dict_file YOUR_DICT_FILE \
--output_dir YOUR_OUTPUT_DIR \
--model_name_or_path PRETRAINED_MODEL \
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




