from peft import LoraConfig, get_peft_model, TaskType
import time
import torch
# from vllm import LLM

def make_peft_config(model_args,data_args):
    return LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, 
    r = model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout
    )

def load_model_and_tokenizer(model_path,tokenizer_path=None,config_path=None,device="cuda"):
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from xenon_generation.models.modeling_gpt2_lab import GPT2LabConfig, GPT2LabLMHeadModel

    if tokenizer_path is None:
        tokenizer_path = model_path
    if config_path is None:
        config_path = model_path
    AutoConfig.register("gpt2lab", GPT2LabConfig)
    AutoModelForCausalLM.register(GPT2LabConfig, GPT2LabLMHeadModel)

    print("Loading Model from {}".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
    config = AutoConfig.from_pretrained(config_path,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,config=config,low_cpu_mem_usage=True,trust_remote_code=True).half().to(device)
    model.eval()
    return tokenizer, model

def generate(model,tokenizer,text,**kwargs):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")  # 注意！如果不删除，会导致每个token加上一个type emb 
    res = model.generate(
        inputs["input_ids"], **kwargs
    )
    return tokenizer.decode(res[0],skip_special_tokens=True)

def generate_batch(model,tokenizer,batch,left_pad,device="cuda",**kwargs):
    input = collate_tokens([b[0] for b in batch],pad_idx=tokenizer.pad_token_id,left_pad=left_pad).to(device)
    encoding = {'input_ids':input}
    completions = []
    with torch.no_grad():
        out_ids = model.generate(
            **encoding,
            **kwargs
        )
        for i, sequence in enumerate(out_ids):
            sequence = sequence.cpu().numpy().tolist()
            completion = tokenizer.decode(
                sequence, skip_special_tokens=True
            )
            completions.append(completion.strip())
    return completions

def generate_batch_vllm(llm,tokenizer,batch,left_pad,sampling_params):
    input = [b[0].tolist() for b in batch]
    completions = []
    with torch.no_grad():
        outputs = llm.generate(
            prompt_token_ids=input,
            sampling_params = sampling_params,
            
        )
        for output in outputs:
            completions.append(output.outputs[0].text)
            
    return completions



def batch_decode(
        token_ids,
        tokenizer
):
        outputs = []
        for _token_ids in token_ids:
            begin = False
            output = []
            for t in _token_ids:
                if  t == tokenizer.eos_token_id:
                    begin = True
                    continue
                elif not begin:
                    continue
                elif t == tokenizer.pad_token_id:
                    outputs.append(tokenizer.convert_tokens_to_string(output))
                    output = []
                    break
                elif t == tokenizer.unk_token_id:
                    continue
                else:
                    output.append(tokenizer._convert_id_to_token(t))
            if len(output) > 0:
                outputs.append(tokenizer.convert_tokens_to_string(output))
        if len(outputs) == 0:
            import pdb; pdb.set_trace()
        return outputs
        



def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def group_to_batches(examples,sizes,max_tokens_per_batch):
    sorted_examples = [e for _,e in sorted(zip(sizes,examples),key=lambda pair: pair[0])]
    sorted_sizes = list(sorted(sizes))
    batches = []
    cur_batch_size = sorted_sizes[0]
    batch = [sorted_examples[0]]
    for idx in range(1,len(examples)):
        if cur_batch_size + sorted_sizes[idx] < max_tokens_per_batch:
            batch.append(sorted_examples[idx])
            cur_batch_size += sorted_sizes[idx]
        else:
            batches.append(batch)
            batch = [sorted_examples[idx]]
            cur_batch_size = sorted_sizes[idx]

    if len(batch) > 0:
        batches.append(batch)
    return batches

class LMPrefixDataLoader:
    def __init__(self,lm_prefixes,tokenizer,max_tokens,max_length=1024):
        datas = []
        sizes = []
        tokenized = tokenizer(lm_prefixes)
        datas = [(torch.tensor(e).long(),i) for i,e in enumerate(tokenized["input_ids"])]
        sizes = [len(e) for e in tokenized["input_ids"]]

        datas,sizes = [],[]
        for i,e in enumerate(tokenized["input_ids"]):
            if len(e) > max_length:
                print("Too long prompt: {}, skipped".format(len(e),max_length))
                continue
            else:
                datas.append((torch.tensor(e[:max_length]).long(),i))
                sizes.append(len(e))
        # for i,lm_prefix in enumerate(lm_prefixes):
        #     input_ids = tokenizer(lm_prefix,return_tensors='pt')['input_ids'].squeeze(0)
        #     datas.append((input_ids,i))        
        #     sizes.append(input_ids.size(0))
        self.batches = group_to_batches(datas,sizes,max_tokens_per_batch=max_tokens)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
    
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
