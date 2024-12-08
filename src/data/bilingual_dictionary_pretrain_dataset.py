from functools import partial
from datasets import load_dataset
import torch
from itertools import chain
from typing import Any, Dict, NewType, Sequence
import transformers

from src.trainer.trainer_utils import collate_tokens

from dataclasses import dataclass, field

@dataclass
class DataCollatorForBDPDataset(object):
    tokenizer: transformers.PretrainedTokenizer

    def __call__(self,instances):
        return_dict = {}
        x_entry = [torch.tensor(instance["x"]).long() for instance in instances]
        y_entry = [torch.tensor(instance["input_ids"]).long() for instance in instances]
        entry = x_entry + y_entry
        data = collate_tokens(entry, self.tokenizer.pad_token_id, left_pad=True)
        return_dict = {
            "input_ids": data,
            "attention_mask": data.ne(self.tokenizer.pad_token_id),
            "labels": data.masked_fill(data.eq(self.tokenizer.pad_token_id),-100)

        }
        return return_dict

def tokenize_parallel_function(tokenizer,examples):
    tokenized_src = tokenizer(examples["src_text"])
    tokenized_tgt = tokenizer(examples["tgt_text"])
    return {
        "x": [s for s in tokenized_src["input_ids"]],
        "input_ids": [t for t in tokenized_tgt["input_ids"]]
    }

def make_bilingual_dict_pretrain_module(data_args,model_args,tokenizer):
    tokenize_parallel_fn = partial(tokenize_parallel_function, tokenizer)

    raw_datasets = load_dataset("json",data_files={"train":data_args.dict_train_file})
    column_names = raw_datasets["train"].column_names

    tokenized_datasets = raw_datasets.map(
        tokenize_parallel_fn,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset"
    )

    train_dataset = tokenized_datasets["train"]

    parallel_data_collator = DataCollatorForBDPDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=train_dataset, data_collator=parallel_data_collator)