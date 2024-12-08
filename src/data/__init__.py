from src.data.lm_dataset import make_lm_data_module
from src.data.lm_with_parallel_dataset import make_lm_with_parallel_module
from src.data.bilingual_dictionary_pretrain_dataset import make_bilingual_dict_pretrain_module
from src.data.bilingula_dictionary_pretrain_data_multi import make_bilingual_dict_pretrain_module_multi
def make_data_module(data_args,model_args,tokenizer,type):
    if type == "lm":
        return make_lm_data_module(data_args,model_args,tokenizer)
    elif type == "lm_with_codeswitch":
        return make_lm_with_parallel_module(data_args,model_args,tokenizer)
    elif type =="bilingual_dict_pretrain":
        return make_bilingual_dict_pretrain_module(data_args,model_args,tokenizer)
    elif type == "bilingual_dict_pretrain_multi":
        return make_bilingual_dict_pretrain_module_multi(data_args,model_args,tokenizer)