from functools import partial
from datasets import load_dataset
import torch
from itertools import chain
from typing import Dict,Sequence
import transformers
import datasets
import random
from collections import defaultdict

from src.trainer.trainer_utils import collate_tokens

from dataclasses import dataclass

def is_empty(fname):
    with open(fname) as f:
        content = f.read()
        return len(content) <= 1

@dataclass
class DataCollatorForLMDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    codeswitch_ratio: float
    codeswitch_table: Dict[tuple,tuple]
    
    def is_start(self,i):
        return self.tokenizer.convert_ids_to_tokens(i).startswith("▁") or i == 0

    def codeswitch(self,input_ids):        
        codeswitched_ids = []
        codeswitched_labels = []

        cur = [input_ids[0]]
        for i in input_ids[1:] + [0]:
            if self.is_start(i):
                # cur word finished
                original_word_seq = cur
                if random.random() < self.codeswitch_ratio and tuple(original_word_seq) in self.codeswitch_table:
                    codeswitch_word_seq = random.choice(self.codeswitch_table[tuple(original_word_seq)])
                    codeswitched_ids.extend(codeswitch_word_seq)
                    codeswitched_labels.extend(original_word_seq[0] + [-100] * len(codeswitch_word_seq[1:]))
                else:
                    codeswitched_ids.extend(original_word_seq)
                    codeswitched_labels.extend(original_word_seq)
                cur = [i]
            else:
                cur.append(i)
        return {
            "codeswitched_ids": codeswitched_ids,
            "codeswitched_labels": codeswitched_labels,
        }

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}

        codeswitched_corpus = instances
        
        codeswitched = [self.codeswitch(instance["input_ids"]) for instance in codeswitched_corpus]
        codeswitched_ids = [torch.tensor(codeswitch["codeswitched_ids"]).long() for codeswitch in codeswitched]
        codeswitched_labels = [torch.tensor(codeswitch["codeswitched_labels"]).long() for codeswitch in codeswitched]
        input_ids = collate_tokens(codeswitched_ids, self.tokenizer.pad_token_id, left_pad=False)
        labels = collate_tokens(codeswitched_labels, -100, left_pad=False)
        return_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": labels,
        }
        return return_dict

def tokenize_function(tokenizer,example):
    return {
        "input_ids": [s+ [tokenizer.eos_token_id] for s in tokenizer(example["text"])["input_ids"]]
    }


def group_texts(block_size,examples):
    # Concatenate all texts.    
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples["input_ids"])
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


            
def build_dataset(files,tokenize_fn,group_fn=None):
    ret_datasets = []
    for file in files:
        if file == "none" or is_empty(file) :
            continue
        dataset_args = {}
        raw_datasets = load_dataset("json", data_files={"train":file}, **dataset_args)
        column_names = raw_datasets["train"].column_names

        dataset = raw_datasets.map(
            tokenize_fn,
            batched=True,
            num_proc=64,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        if group_fn is not None:
            dataset = dataset.map(
                group_fn,
                batched=True,
                num_proc=64,
                load_from_cache_file=True,
                desc="Grouping texts in chunks of 1024"
            )

        ret_datasets.append(dataset["train"])

    if len(ret_datasets) == 0:
        print("No available content from {}".format(files))
        return None
    else:
        return datasets.concatenate_datasets(ret_datasets)


def make_lm_with_parallel_module(data_args,model_args,tokenizer):
    tokenize_fn = partial(tokenize_function,tokenizer)
    group_fn = partial(group_texts,model_args.max_position_embeddings)

    lm_datasets = build_dataset(data_args.lm_train_file.split(","),tokenize_fn,group_fn)
    
    if data_args.codeswitch_ratio > 0:
        codeswitch_table = load_codeswitch_table_from_file(data_args.dict_file,tokenizer)
    else:
        codeswitch_table = None

    lm_data_collator = DataCollatorForLMDataset(tokenizer=tokenizer,codeswitch_table=codeswitch_table,codeswitch_ratio=data_args.codeswitch_ratio,codeswitch_corpus_ratio=data_args.codeswitch_corpus_ratio)

    return dict(train_dataset=lm_datasets, eval_dataset=None, data_collator=lm_data_collator)


def load_codeswitch_table_from_file(dict_file,tokenizer):
    print("Loading dict from {}".format(dict_file))
    codeswitch_table = defaultdict(lambda: [])
    with open(dict_file) as f:
        for line in f:
            src_word, tgt_word = tuple(line.strip().split("\t"))
            src_word_seq = tuple(tokenizer(src_word)["input_ids"])
            tgt_word_seq = tokenizer(tgt_word)["input_ids"]
            codeswitch_table[src_word_seq].append(tgt_word_seq)
    return codeswitch_table


