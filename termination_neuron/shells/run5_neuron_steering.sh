# ================================================
# run1_complete.py
# ================================================
export OMP_NUM_THREADS=8
model_names=(
    # google/gemma-2-2b
    # google/gemma-3-4b-it
    # meta-llama/Meta-Llama-3.1-8B-Instruct
    # Qwen/Qwen3-4B-Instruct-2507
    meta-llama/Meta-Llama-3.1-8B-Instruct


    # kakaocorp/kanana-1.5-8b-instruct-2505
    # LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
    # google/gemma-3-12b-it
    # Qwen/Qwen3-4B-Thinking-2507
    # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
    # kakaocorp/kanana-1.5-2.1b-instruct-2505
)

# 11 
neuron_paths=(
    run2_incomplete_comma
    run2_incomplete_continue
    run2_incomplete_example
    run3_incomplete_comma_termination
    run3_incomplete_continue_termination
    run3_incomplete_example_termination
    run3_incomplete_comma_jailbreak_termination
    run3_incomplete_continue_jailbreak_termination
    run3_incomplete_example_jailbreak_termination
    run3_incomplete_comma_non_termination
    run3_incomplete_continue_non_termination
    run3_incomplete_example_non_termination
)
augment_types=(
    # "comma"
    "continue"
    # "example"
)
at=1
threshold=0.25
batch_size=4
magnitude_alphas=(
    # 0.25
    # 2.0 
    4.0
)

seed=42
for augment_type in "${augment_types[@]}"; do
for model_name in "${model_names[@]}"; do
for neuron_path in "${neuron_paths[@]}"; do
for magnitude_alpha in "${magnitude_alphas[@]}"; do
    echo "Running 4, Steering,  $model_name $augment_type $neuron_path" 
    # python run5_neuron_steering.py \
    #     --model_name $model_name \
    #     --batch_size $batch_size \
    #     --neuron_path $neuron_path \
    #     --at $at \
    #     --threshold $threshold \
    #     --augment_type $augment_type \
    #     --magnitude_alpha $magnitude_alpha
    
    evaluation_model_name=gpt-4o-mini
    python run0_evaluate_by_openai.py \
        --model_name $model_name \
        --eval_model_name $evaluation_model_name \
        --eval_task "run5_neuron_steering/"$neuron_path"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
        --seed $seed \
        --openai_api_key $OPENAI_API_KEY

    python run0_evaluate_normality_by_openai.py \
        --model_name $model_name \
        --eval_model_name $evaluation_model_name \
        --eval_task "run5_neuron_steering/"$neuron_path"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
        --seed $seed \
        --openai_api_key $OPENAI_API_KEY


    # evaluation_model_name=google/gemma-3-12b-it
    # python run0_evaluate_by_llm.py \
    #     --model_name $model_name \
    #     --eval_model_name $evaluation_model_name \
    #     --eval_task "run5_neuron_steering/"$neuron_path"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
    #     --seed $seed \
    #     --batch_size 2

    # python run0_evaluate_normality.py \
    #     --model_name $model_name \
    #     --eval_model_name $evaluation_model_name \
    #     --eval_task "run5_neuron_steering/"$neuron_path"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
    #     --seed $seed \
    #     --batch_size 2

done 
done 
done
done

# ================================================