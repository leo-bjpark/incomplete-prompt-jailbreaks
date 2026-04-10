# ================================================
# run1_complete.py
# ================================================
export OMP_NUM_THREADS=8
model_names=(
    google/gemma-2-2b
    google/gemma-3-4b-it
    meta-llama/Meta-Llama-3.1-8B-Instruct
    Qwen/Qwen3-4B-Instruct-2507

    # kakaocorp/kanana-1.5-8b-instruct-2505
    # LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
    # google/gemma-3-12b-it
    # Qwen/Qwen3-4B-Thinking-2507
    # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
    # kakaocorp/kanana-1.5-2.1b-instruct-2505
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
    0.25
    2.0 
    4.0
)

# ================================================
# Remove Termination From Jailbreak + Termination
neuron_path=run3_incomplete_continue_jailbreak_termination 
neuron_path_neg=run3_incomplete_continue_termination
# ================================================

# # # ================================================
# # # Remove Non-Termination From Jailbreak + Termination Detection Neuron
neuron_path=run3_incomplete_continue_jailbreak_termination 
neuron_path_neg=run3_incomplete_continue_non_termination
# # # ================================================

# # # ================================================
# # # Remove Jailbreak + Termination From Non-Termination 
# neuron_path=run3_incomplete_continue_non_termination 
# neuron_path_neg=run3_incomplete_continue_jailbreak_termination
# # # ================================================

seed=42
for augment_type in "${augment_types[@]}"; do
for model_name in "${model_names[@]}"; do
for magnitude_alpha in "${magnitude_alphas[@]}"; do
    echo "Running 6, Steering,  $model_name $augment_type $neuron_path" 
    # python run6_neuron_steering_pos_and_neg.py \
    #     --model_name $model_name \
    #     --batch_size $batch_size \
    #     --neuron_path $neuron_path \
    #     --at $at \
    #     --threshold $threshold \
    #     --augment_type $augment_type \
    #     --magnitude_alpha $magnitude_alpha \
    #     --neuron_path_neg $neuron_path_neg


    # evaluation_model_name=gpt-4o-mini
    # python run0_evaluate_by_openai.py \
    #     --model_name $model_name \
    #     --eval_model_name $evaluation_model_name \
    #     --eval_task "run6_neuron_steering_pos_and_neg/"$neuron_path"/"$neuron_path_neg"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
    #     --seed $seed \
    #     --openai_api_key $OPENAI_API_KEY

    # python run0_evaluate_normality_by_openai.py \
    #     --model_name $model_name \
    #     --eval_model_name $evaluation_model_name \
    #     --eval_task "run6_neuron_steering_pos_and_neg/"$neuron_path"/"$neuron_path_neg"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
    #     --seed $seed \
    #     --openai_api_key $OPENAI_API_KEY

    evaluation_model_name=google/gemma-3-12b-it
    # python run0_evaluate_by_llm.py \
    #     --model_name $model_name \
    #     --eval_model_name $evaluation_model_name \
    #     --eval_task "run6_neuron_steering_pos_and_neg/"$neuron_path"/"$neuron_path_neg"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
    #     --seed $seed \
    #     --batch_size 2

    python run0_evaluate_normality.py \
        --model_name $model_name \
        --eval_model_name $evaluation_model_name \
        --eval_task "run6_neuron_steering_pos_and_neg/"$neuron_path"/"$neuron_path_neg"/at_"$at"_threshold_"$threshold"_magnitude_alpha_"$magnitude_alpha \
        --seed $seed \
        --batch_size 2





done 
done
done

# ================================================