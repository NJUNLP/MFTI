set -e

plm=${plm:-""}
train_file=${train_file:-""}
valid_file=${valid_file:-""}
batch_size=${batch_size:-"10"}
max_length=${max_length:-"256"}
update_freq=${update_freq:-"1"}
savedir=${savedir:-""}
lr=${lr:-"5e-6"}
ds_config=${ds_config:-"configs/stage2.json"}
devices=${devices:-"0,1,2,3,4,5,6,7"}
master_port=${master_port:-"22222"}
save_steps=${save_steps:-"250"}
max_epochs=${max_epochs:-"1"}
max_steps=${max_steps:-"2000"}
model_type=${model_type:-"vanilla"}
#align args
align_alpha=${align_alpha:-"0"}
# peft args
peft_type=${peft_type:-"lora"}
lora_r=${lora_r:-"4"}
peft_per=${peft_per:-"language"}
use_language_id=${use_language_id:-"False"}
use_shared_lora=${use_shared_lora:-"False"}
lora_composition_type=${lora_composition_type:-"multiply"}
template=${template:-"1"}
flores101_dataset=${flores101_dataset:-}

# evaluate
validation_data=${validation_data:-""}
test_data=${test_data:-""}

blocking_ngram=${blocking_ngram:""}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done


local_train_file=$train_file

local_plm=$plm



local_flores_data=flores101_dataset

if [[ $blocking_ngram == "" ]];then
    blocking_ngram_args=""
else
    blocking_ngram_args="--blocking_ngram ${blocking_ngram}"
fi

if [[ $ds_config == "" ]];then
    ds_config_args=""
else
    ds_config_args="--deepspeed $ds_config"
fi

if [[ $max_steps != "" ]];then
    max_args="--max_steps $max_steps"
else
    max_args="--num_train_epochs $max_epochs"
fi


deepspeed --master_port $master_port --include="localhost:${devices}" instruct_mt/run_clm_with_lora_3.py \
    --model_name_or_path $local_plm \
    --train_file $local_train_file \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --group_by_length --model_max_length $max_length\
    --gradient_accumulation_step $update_freq \
    --fp16 --report_to none\
    --do_train --remove_unused_column False\
    --output_dir $savedir \
    $max_args --logging_steps 50\
    --evaluation_strategy no \
    --save_strategy steps --save_steps $save_steps\
    --learning_rate $lr $ds_config_args\
    --peft_type $peft_type --lora_r $lora_r --peft_per $peft_per $blocking_ngram_args \
    --use_language_id $use_language_id \
    --model_type $model_type --lora_composition_type $lora_composition_type --use_shared_lora $use_shared_lora \
    --align_alpha $align_alpha


python3 -m instruct_mt.evaluate_clm_with_language_id\
     --model-name-or-path $savedir \
     --validation-file $local_flores_data/dev/valid \
     --test-file $local_flores_data/test/test \
      --max-tokens 512 \
      --tokenize flores101 \
      --savedir $LOCAL_OUTPUT \
      --save-steps 250 \
      --template $template