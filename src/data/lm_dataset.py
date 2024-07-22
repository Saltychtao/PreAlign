from functools import partial
from datasets import load_dataset
from accelerate import Accelerator
import torch
from itertools import chain
from typing import Any,  Dict, NewType,Sequence
InputDataClass = NewType("InputDataClass", Any)
import transformers

from dataclasses import dataclass, field

@dataclass
class DataCollatorForSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return_dict = {}
        for key in ("input_ids","labels"):
            if key not in instances[0]:
                continue
            entry = [torch.tensor(instance[key]).long() for instance in instances]
            data = torch.nn.utils.rnn.pad_sequence(
                entry, batch_first=True, padding_value=self.tokenizer.pad_token_id,)
            if "labels" in key:
                data[data.eq(self.tokenizer.pad_token_id)] = -100
            return_dict[key] = data
        return_dict["attention_mask"] = return_dict["input_ids"].ne(self.tokenizer.pad_token_id)
        return return_dict

def tokenize_function(tokenizer,examples):
    output = tokenizer(examples["text"])
    return output


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
            

def make_lm_data_module(data_args,model_args,tokenizer):
    tokenize_fn = partial(tokenize_function,tokenizer)
    group_fn = partial(group_texts,model_args.max_position_embeddings)
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = data_args.train_file.split(".")[-1]
    import pdb; pdb.set_trace()
    raw_datasets = load_dataset(extension, data_files=data_files,**dataset_args)
    column_names = raw_datasets["train"].column_names


    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=60,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    lm_datasets = tokenized_datasets.map(
        group_fn,
        batched=True,
        num_proc=120,
        load_from_cache_file=True,
        desc="Grouping texts in chunks of 1024"
    )
    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
 