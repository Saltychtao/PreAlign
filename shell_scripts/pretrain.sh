set -e

train_file=${train_file:-""}
output_dir=${output_dir:-""}

model_name_or_path=${model_name_or_path:-""}
dict_file=${dict_file:-""}
codeswitch_ratio=${codeswitch_ratio:-""}
tokenizer_path=${tokenizer_path:-"/mnt/bn/st-data-lq/jiahuanli/models/Baichuan2-13B-base"}

master_port=${master_port:-"2222"}

devices=${devices:-""}
ds_config=${ds_config:-"configs/stage1.json"}


while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

pip3 install flash-attn --no-build-isolation

WANDB_PROJECT=${project} WANDB_NAME=${expr} deepspeed --master_port $master_port --include="localhost:$devices" src/pretrain.py \
    --model_name_or_path $model_name_or_path --tokenizer_path $model_name_or_path \
    --train_file $train_file --dict_file $dict_file \
    --bf16 --do_train \
    --output_dir $output_dir \
    --seed 42 --report_to none\ 
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4\
    --learning_rate 2e-4 \
    --codeswitch_ratio ${codeswitch_ratio}
 