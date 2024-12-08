from functools import partial
from datasets import load_dataset, load_from_disk
import datasets
from datasets import concatenate_datasets
from accelerate import Accelerator
import torch
from itertools import chain
from typing import Any,  Dict, NewType,Sequence
InputDataClass = NewType("InputDataClass", Any)
import transformers
from transformers import DataCollatorForLanguageModeling

from dataclasses import dataclass, field

from collections import defaultdict
from tqdm import tqdm

from src.trainer.trainer_utils import collate_tokens
import random

@dataclass
class DataCollatorForLMDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    codeswitch_ratio: float
    codeswitch_table: Dict[tuple,tuple]
    codeswitch_corpus_ratio: float

    def is_start(self,i):
        if i >= self.tokenizer:
            i -= len(self.tokenizer)
        return self.tokenizer.convert_ids_to_tokens(i).startswith("_") or self.tokenizer.convert_ids_to_tokens(i).startswith("") or i == 0

    def codeswitch(self, input_ids):
        codeswitched_id = []
        codeswitched_align_mask = []
        original_align_mask = []
        cur = [input_ids[0]]
        for i in input_ids[1:] + [0]:
            if self.tokenizer.convert_ids_to_tokens(i).startswith("_") or self.tokenizer.convert_ids_to_tokens(i).startswith("TODO") or i == 0:
                original_word_seq = cur
                if random.random() < self.codeswitch_ratio and tuple(original_word_seq) in self.codeswitch_table:
                    codeswitch_word_seq = random.choice(self.codeswitch_table[tuple(original_word_seq)])
                    codeswitch_ids.extend(codeswitch_word_seq)
                else:
                    codeswitched_ids.extend(original_word_seq)
                    codeswitched_align_mask.extend([1]*len(original_word_seq))
                    original_align_mask.extend([1]*(len(original_word_seq)))
                cur = [i]
            else:
                cur.append(i)
        return {
            "codeswitched_ids": codeswitched_ids,
            "original_ids": input_ids
        }


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        codeswitched =[self.codeswitch(instance["input_ids"]) for instance in instances]
        codeswitched_ids = [torch.tensor(codeswitch["codeswitched_ids"]).long() for codeswitch in codeswitched]

        input_ids = collate_tokens(codeswitched_ids, self.tokenizer.pad_token_id, left_pad=False)

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": input_ids.masked_fill(input_ids.eq(self.tokenizer.pad_token_id),-100)
        }
        return return_dict

def tokenize_function(tokenizer,examples):
    output = tokenizer(examples["text"])
    return {
        "input_ids": [i + [tokenizer.eos_token_id] for i in output["input_ids"]]
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

def build_dataset(data_file, tokenize_fn, group_fn, subset_ratio):
    raw_dataset = load_from_disk(data_file)
    tokenized_datasets = raw_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=60,
        remove_columns=raw_dataset.column_names,
        desc="Running Tokenizer on dataset"
    )
    lm_dataset = tokenized_datasets.map(
        group_fn,
        batched=True,
        num_proc=60,
        desc="Grouping texts in chunks of 1024"
    )

    return lm_dataset
            
def load_codeswitch_table(tokenizer,dict_file):
    import json
    dict_data = []
    with open(dict_file) as f:
        for line in f:
            dict_data.append(json.loads(line))

    codeswitch_table = defaultdict(list)
    for d in tqdm(dict_data):
        words = d["words"].split(" $ ")
        src = words[0]
        src_tokenized = tokenizer(src)["input_ids"]
        if tokenizer.convert_ids_to_tokens(src_tokenized[0]) in tokenizer.special_tokens_map.values():
            src_tokenized = src_tokenized[1:]
        tgt = words[1:]
        for tgt in tgts:
            tgt_tokenized = tokenizer(tgt)["input_ids"]
            if tokenizer.convert_ids_to_tokens(tgt_tokenized[0]) in tokenizer.special_tokens_map.values():
                tgt_tokenized = tgt_tokenized[1:]
            codeswitch_table[tuple(src_tokenized)].append(tgt_tokenized)
        return codeswitch_table

def make_lm_data_module(data_args,model_args,tokenizer):
    tokenize_fn = partial(tokenize_function,tokenizer)
    group_fn = partial(group_texts,model_args.max_position_embeddings)
    train_dataset = build_dataaset(data_args.train_file, tokenize_fn,group_fn,data_args.subset_ratio)
    valid_datasets = build_validation_dataset(data_args.validation_file, tokenize_fn, group_fn)
    if data_args.codeswitch_ratio == 0:
        data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)
    else:
        codeswitch_table = torch.load(data_args.codeswitch_table_file)
        data_collator = DataCollatorForLMDataset(tokenizre,data_args.codeswitch_ratio,codeswitch_table,1)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
 