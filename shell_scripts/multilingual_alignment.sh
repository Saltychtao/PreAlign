set -e

lm_train_file=${lm_train_file:-""}
output_dir=${output_dir:-""}

tokenizer_path=${tokenizer_path:-""}

master_port=${master_port:-"2222"}

devices=${devices:-""}
ds_config=${ds_config:-"configs/deepspeed/stage1.json"}

dict_train_file=${dict_train_file:-""}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done


deepspeed --master_port $master_port --include="localhost:$devices" src/bilingual_dictionary_pretrain.py \
    --model_name_or_path $model_name_or_path \
    --remove_unused_columns False\
    --bf16 --do_train  --dataloader_num_workers 0\
    --output_dir $output_dir \
    --seed 42 --report_to none\
    --tokenizer_path $tokenizer_path \
    --deepspeed $ds_config \
    --dict_train_file ${dict_train_file} \
    --output_dir ${output_dir} \
    --contrastive_multi 1