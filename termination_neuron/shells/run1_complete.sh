# ================================================
# run1_complete.py
# ================================================
export OMP_NUM_THREADS=8
model_names=(
    meta-llama/Meta-Llama-3.1-8B-Instruct
    # google/gemma-2-2b
    # google/gemma-3-4b-it
    # kakaocorp/kanana-1.5-8b-instruct-2505
    # LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
    # Qwen/Qwen3-4B-Instruct-2507
    # google/gemma-3-12b-it
    # Qwen/Qwen3-4B-Thinking-2507
    # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
    # kakaocorp/kanana-1.5-2.1b-instruct-2505
)

evaluation_model_name=google/gemma-3-12b-it

seed=42
batch_size=8
for model_name in "${model_names[@]}"; do
echo "Running 1, Complete Prompting,  $model_name" 
python run1_complete.py \
    --model_name $model_name \
    --seed $seed \
    --batch_size $batch_size

python run0_evaluate_by_llm.py \
    --model_name $model_name \
    --eval_model_name $evaluation_model_name \
    --eval_task "run1_complete_prompting" \
    --seed $seed \
    --batch_size 2
done
# ================================================