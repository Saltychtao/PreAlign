from src.data.lm_dataset import make_lm_data_module
from src.data.lm_with_parallel_dataset import make_lm_with_parallel_module

def make_data_module(data_args,model_args,tokenizer,type):
    if type == "lm":
        return make_lm_data_module(data_args,model_args,tokenizer)
    elif type == "lm_with_codeswitch":
        return make_lm_with_parallel_module(data_args,model_args,tokenizer)