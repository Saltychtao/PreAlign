set -e

train_file=${train_file:-""}
output_dir=${output_dir:-""}

hf_config=${hf_config:-"/mnt/bn/st-data-lq/jiahuanli/models/Baichuan2-13B-base"}
config_file=${config_file:-""}

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
    --config_file $config_file \
    --train_file $train_file \
    --validation_file $validation_file \
    --fp16 --do_train \
    --hf_config $hf_config \
    --output_dir $output_dir \
    --seed 42 --report_to none \
    --tokenizer_path $tokenizer_path