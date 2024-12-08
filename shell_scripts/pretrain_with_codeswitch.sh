set -e

lm_train_file=${lm_train_file:-""}
output_dir=${output_dir:-""}

hf_config=${hf_config:-""}
config_file=${config_file:-""}
tokenizer_path=${tokenizer_path:-""}

master_port=${master_port:-"2222"}

devices=${devices:-""}
ds_config=${ds_config:-"configs/deepspeed/stage1.json"}

dict_file=${dict_file:-""}
codeswitch_ratio=${codeswitch_ratio:-"0"}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done


deepspeed --master_port $master_port --include="localhost:$devices" src/pretrain.py \
    --config_file $config_file \
    --lm_train_file $lm_train_file \
    --fp16 --do_train  --dataloader_num_workers 0\
    --hf_config $hf_config \
    --output_dir $output_dir \
    --seed 42 --report_to none\
    --tokenizer_path $tokenizer_path \
    --deepspeed $ds_config \
    --dict_file ${dict_file} \
    --codeswitch_ratio $codeswitch_ratio 
