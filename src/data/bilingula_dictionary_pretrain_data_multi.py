from functools import partial
from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch
from itertools import chain
from typing import Any, Dict, NewType, Sequence
import transformers


from src.trainer.trainer_utils import collate_tokens

from dataclasses import dataclass

@dataclass
class DataCollatorForBDPDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        return_dict = {}
        dict_entry = [torch.tensor(instance["input_ids"]).long() for instance in instances]
        dict_indices = [torch.tensor(instance["indices"]).long() for instance in instances]
        dict_data = collate_tokens(dict_entry, self.tokenizer.pad_token_id,left_pad=False)
        dict_indices = collate_tokens(dict_indices,-100,left_pad=False)
        B,k = dict_indices.size()
        dict_indices = dict_indices.view(B*k)
        dict_data = dict_data.view(B*k,-1)

        return_dict = {
            "input_ids": dict_data,
            "attention_mask": dict_data.ne(self.tokenizer.pad_token_id),
            "labels": dict_data.masked_fill(dict_data.eq(self.tokenizer.pad_token_id)),
            "indices": dict_indices
        }
        return return_dict

def pad_and_concatenate(xs,target_length,sep_id, pad_token_idx):
    padded_sublists = []

    current_sublist = []

    for x in xs:
        if x == sep_id:
            current_sublist = [pad_token_idx] * (target_length-len(current_sublist)) + current_sublist
            padded_sublists.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.append(x)
    if current_sublist:
        current_sublist = [pad_token_idx] * (target_length - len(current_sublist)) + current_sublist
        padded_sublists.append(current_sublist)
    concatenated_list = [item for sublist in padded_sublists for item in sublist]
    return concatenated_list, len(padded_sublists)

def tokenize_parallel_function(tokenizer,examples):
    tokenized = tokenizer(examples["words"])["input_ids"]
    indices = examples["index"]
    sequences = []
    numbers = []
    if "llama" in tokenizer.name_or_path.lower():
        sep_id = 395
    elif "baichuan" in tokenizer.name_or_path.lower():
        sep_id = 1071
    elif "gpt" in tokenizer.name_or_path.lower():
        sep_id = 400
    elif "qwen" in tokenizer.name_or_path.lower():
        sep_id = 400
    elif "pythia" in tokenizer.name_or_path.lower():
        sep_id = 370
    for x in tokenized:
        concatenated_list, n= pad_and_concatenate(x,30,sep_id, tokenizer.pad_token_id)
        sequences.append(concatenated_list)
        numbers.append(n)

    return {
        "input_ids": sequences,
        "indices": [[i]*n for i,n in zip(indices,numbers)]
    }

def group_texts(block_size,examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[1]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def make_bilingual_dict_pretrain_module_multi(data_args, model_args,tokenizer):
    tokenize_parallel_fn = partial(tokenize_parallel_function, tokenizer)

    raw_datasets = load_dataset("json", data_files={"train":data_args.dict_train_file})["train"]
    column_names = raw_datasets.column_names
    dict_dataset = raw_datasets.map(
        tokenize_parallel_fn,
        batched=True,
        num_proc=64,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running Tokenizer on dataset"
    )

    parallel_data_collator = DataCollatorForBDPDataset(tokenizer=tokenizer)

    return dict(train_dataset=dict_dataset,eval_dataset=None, data_collator=parallel_data_collator)