# ================================================
# run1_complete.py
# ================================================
export OMP_NUM_THREADS=8
model_names=(
    # google/gemma-2-2b
    meta-llama/Meta-Llama-3.1-8B-Instruct
    # google/gemma-3-4b-it
    # kakaocorp/kanana-1.5-8b-instruct-2505
    # LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
    # Qwen/Qwen3-4B-Instruct-2507


    # google/gemma-3-12b-it
    # Qwen/Qwen3-4B-Thinking-2507
    # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
    # kakaocorp/kanana-1.5-2.1b-instruct-2505
)

data_exp_names=(
    # run2_incomplete_comma
    # run2_incomplete_continue
    # run2_incomplete_example
    # run3_incomplete_comma_termination
    # run3_incomplete_continue_termination
    # run3_incomplete_example_termination
    # run3_incomplete_comma_jailbreak_termination
    # run3_incomplete_continue_jailbreak_termination
    # run3_incomplete_example_jailbreak_termination
    # run3_incomplete_comma_non_termination
    # run3_incomplete_continue_non_termination
    # run3_incomplete_example_non_termination
)

batch_size=4
for model_name in "${model_names[@]}"; do
for data_exp_name in "${data_exp_names[@]}"; do
    echo "Running 4, Termination Neuron Detection,  $model_name" 
    python run4_termination_neuron_detection.py \
        --model_name $model_name \
        --batch_size $batch_size \
        --data_exp_name $data_exp_name \

done 
done
# ================================================