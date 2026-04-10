import os 
import json
import torch
import argparse
import gc
import psutil
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
parser.add_argument("--batch_size", type=int, default=8)  # Reduced default batch size
parser.add_argument("--max_memory_gb", type=float, default=20.0)  # Maximum memory usage in GB
parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate results to disk")
parser.add_argument("--random_sample_ratio", type=float, default=1.0, help="Ratio of random sampling for non-end-symbol positions (0.0-1.0)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
parser.add_argument("--data_exp_name", type=str, default="run3_incomplete_comma", help="Experiment name")

args = parser.parse_args()

args.save_dir = f"outputs/run4_termination_neuron_detection/{args.data_exp_name}/{args.model_name}"

# Set random seed for reproducible sampling
torch.manual_seed(args.seed)
import random
random.seed(args.seed)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def check_memory_limit():
    """Check if memory usage exceeds limit"""
    current_memory = get_memory_usage()
    if current_memory > args.max_memory_gb:
        print(f"Warning: Memory usage ({current_memory:.2f} GB) exceeds limit ({args.max_memory_gb} GB)")
        return True
    return False

def cleanup_memory():
    """Clean up memory by forcing garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# ================================================
experiment_settings = [
    'run3_incomplete_comma',
    'run3_incomplete_example',
    'run3_incomplete_continue',
]

termination_prompt_types = [
    "jailbreak_termination",
    "termination",
    "non_termination",
]

exp_names = [
    # 'run2_incomplete_prompting',
    # 'run2_incomplete_comma',
    # 'run2_incomplete_continue',
    # 'run2_incomplete_example',    
]

if 'run2' in args.data_exp_name:
    exp_names = [args.data_exp_name]
else: 
    termination_prompt_versions = [1,2,3]
    for termination_prompt_version in termination_prompt_versions:
        exp_names.append(f"{args.data_exp_name}_{termination_prompt_version}")


gathered_generations = []
for exp_name in exp_names:
    path = f"outputs/{exp_name}/{args.model_name}/all_outputs.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            for sample in data:
                gathered_generations.append(sample['output_text'])


model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                             device_map='auto',
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

def custom_collate_fn_with_end_symbol_indices(batch, tokenizer):
    texts = [x['text'] for x in batch]
    tokenized = tokenizer(texts, 
                          padding=True,            # longest in batch
                          truncation=False,        # preserve all information
                          return_tensors="pt", 
                          return_attention_mask=True,
                          )
    end_symbols = [".", "!", "?",]
    def get_last_sentence(token_ids):
        # token_ids = tokenizer.encode(text, add_special_tokens=False)
        end_symbol_indices = []
        for i, tid in enumerate(token_ids):
            # 한 토큰만 디코드 + 공백 정리 끄기 (줄바꿈 보존)
            decoded_tok = tokenizer.decode(
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )
            if any(sym in decoded_tok for sym in end_symbols):
                end_symbol_indices.append(i)
        return end_symbol_indices
    
    tokenized['end_symbol_indices'] = []
    for i in range(len(tokenized['input_ids'])):
        token_ids = tokenized['input_ids'][i]
        end_symbol_indices = get_last_sentence(token_ids)
        tokenized['end_symbol_indices'].append(end_symbol_indices)

    return tokenized    


class HiddenStateHook:
    def __init__(self):
        self.hidden_states_at_0 = None
        self.hidden_states_at_1 = None
        self.hidden_states_at_2 = None
        self.hidden_states_at_3 = None
        
        self.rest_of_indices_at_0 = None
        self.rest_of_indices_at_1 = None
        self.rest_of_indices_at_2 = None
        self.rest_of_indices_at_3 = None
        
        self.end_symbol_indices = None
        
    def __call__(self, module, input, output):
        # Pre-allocate lists to avoid repeated memory allocation
        batch_size = input[0].shape[0]
        seq_len = input[0].shape[1]
        
        # Initialize lists with None to avoid memory fragmentation
        self.hidden_states_at_0 = []
        self.hidden_states_at_1 = []
        self.hidden_states_at_2 = []
        self.hidden_states_at_3 = []
        
        self.rest_of_indices_at_0 = []
        self.rest_of_indices_at_1 = []
        self.rest_of_indices_at_2 = []
        self.rest_of_indices_at_3 = []
        
        # Pre-compute all possible indices to avoid repeated computation
        all_indices = torch.arange(seq_len, device=input[0].device)
        
        for i, end_symbol_index in enumerate(self.end_symbol_indices):
            if len(end_symbol_index) == 0:
                continue
                
            end_symbol_index = torch.tensor(end_symbol_index, device=input[0].device)
            
            # Compute all offset indices at once
            end_symbol_index_at_0 = end_symbol_index
            end_symbol_index_at_1 = end_symbol_index - 1 
            end_symbol_index_at_2 = end_symbol_index - 2
            end_symbol_index_at_3 = end_symbol_index - 3 
            
            # Filter valid indices (>= 0)
            valid_mask_0 = end_symbol_index_at_0 >= 0
            valid_mask_1 = end_symbol_index_at_1 >= 0
            valid_mask_2 = end_symbol_index_at_2 >= 0
            valid_mask_3 = end_symbol_index_at_3 >= 0
            
            end_symbol_index_at_0 = end_symbol_index_at_0[valid_mask_0]
            end_symbol_index_at_1 = end_symbol_index_at_1[valid_mask_1]
            end_symbol_index_at_2 = end_symbol_index_at_2[valid_mask_2]
            end_symbol_index_at_3 = end_symbol_index_at_3[valid_mask_3]
            
            # Compute rest indices more efficiently
            rest_of_indices_at_0 = all_indices[~torch.isin(all_indices, end_symbol_index_at_0)]
            rest_of_indices_at_1 = all_indices[~torch.isin(all_indices, end_symbol_index_at_1)]
            rest_of_indices_at_2 = all_indices[~torch.isin(all_indices, end_symbol_index_at_2)]
            rest_of_indices_at_3 = all_indices[~torch.isin(all_indices, end_symbol_index_at_3)]
            
            # Calculate how many samples we need (match end_symbol count)
            target_count_0 = int(len(end_symbol_index_at_0) * args.random_sample_ratio)
            target_count_1 = int(len(end_symbol_index_at_1) * args.random_sample_ratio)
            target_count_2 = int(len(end_symbol_index_at_2) * args.random_sample_ratio)
            target_count_3 =  int(len(end_symbol_index_at_3) * args.random_sample_ratio)
            
            # Random sampling
            if len(rest_of_indices_at_0) > target_count_0:
                perm = torch.randperm(len(rest_of_indices_at_0))
                rest_of_indices_at_0 = rest_of_indices_at_0[perm[:target_count_0]]
            if len(rest_of_indices_at_1) > target_count_1:
                perm = torch.randperm(len(rest_of_indices_at_1))
                rest_of_indices_at_1 = rest_of_indices_at_1[perm[:target_count_1]]
            if len(rest_of_indices_at_2) > target_count_2:
                perm = torch.randperm(len(rest_of_indices_at_2))
                rest_of_indices_at_2 = rest_of_indices_at_2[perm[:target_count_2]]
            if len(rest_of_indices_at_3) > target_count_3:
                perm = torch.randperm(len(rest_of_indices_at_3))
                rest_of_indices_at_3 = rest_of_indices_at_3[perm[:target_count_3]]
        
            # Extract hidden states and convert to half precision immediately
            if len(end_symbol_index_at_0) > 0:
                self.hidden_states_at_0.append(input[0][i, end_symbol_index_at_0, :].detach().cpu().half())
            if len(end_symbol_index_at_1) > 0:
                self.hidden_states_at_1.append(input[0][i, end_symbol_index_at_1, :].detach().cpu().half())
            if len(end_symbol_index_at_2) > 0:
                self.hidden_states_at_2.append(input[0][i, end_symbol_index_at_2, :].detach().cpu().half())
            if len(end_symbol_index_at_3) > 0:
                self.hidden_states_at_3.append(input[0][i, end_symbol_index_at_3, :].detach().cpu().half())
            
            # Extract rest indices
            if len(rest_of_indices_at_0) > 0:
                self.rest_of_indices_at_0.append(input[0][i, rest_of_indices_at_0, :].detach().cpu().half())
            if len(rest_of_indices_at_1) > 0:
                self.rest_of_indices_at_1.append(input[0][i, rest_of_indices_at_1, :].detach().cpu().half())
            if len(rest_of_indices_at_2) > 0:
                self.rest_of_indices_at_2.append(input[0][i, rest_of_indices_at_2, :].detach().cpu().half())
            if len(rest_of_indices_at_3) > 0:
                self.rest_of_indices_at_3.append(input[0][i, rest_of_indices_at_3, :].detach().cpu().half())
        
        # Concatenate only if lists are not empty - more memory efficient
        self.hidden_states_at_0 = torch.cat(self.hidden_states_at_0, dim=0) if self.hidden_states_at_0 else torch.tensor([], dtype=torch.float16)
        self.hidden_states_at_1 = torch.cat(self.hidden_states_at_1, dim=0) if self.hidden_states_at_1 else torch.tensor([], dtype=torch.float16)
        self.hidden_states_at_2 = torch.cat(self.hidden_states_at_2, dim=0) if self.hidden_states_at_2 else torch.tensor([], dtype=torch.float16)
        self.hidden_states_at_3 = torch.cat(self.hidden_states_at_3, dim=0) if self.hidden_states_at_3 else torch.tensor([], dtype=torch.float16)
        
        self.rest_of_indices_at_0 = torch.cat(self.rest_of_indices_at_0, dim=0) if self.rest_of_indices_at_0 else torch.tensor([], dtype=torch.float16)
        self.rest_of_indices_at_1 = torch.cat(self.rest_of_indices_at_1, dim=0) if self.rest_of_indices_at_1 else torch.tensor([], dtype=torch.float16)
        self.rest_of_indices_at_2 = torch.cat(self.rest_of_indices_at_2, dim=0) if self.rest_of_indices_at_2 else torch.tensor([], dtype=torch.float16)
        self.rest_of_indices_at_3 = torch.cat(self.rest_of_indices_at_3, dim=0) if self.rest_of_indices_at_3 else torch.tensor([], dtype=torch.float16)
        
        
    def clear(self):
        """Clear all stored hidden states to free memory"""
        self.hidden_states_at_0 = None
        self.hidden_states_at_1 = None
        self.hidden_states_at_2 = None
        self.hidden_states_at_3 = None
        self.rest_of_indices_at_0 = None
        self.rest_of_indices_at_1 = None
        self.rest_of_indices_at_2 = None
        self.rest_of_indices_at_3 = None


def get_llm_block(llm, llm_name):
    if llm_name == "gpt2":
        block = llm.transformer.h
    elif 'meta-llama' in llm_name:
        block = llm.model.layers
    elif 'Qwen' in llm_name:
        block = llm.model.layers
    elif 'kakaocorp' in llm_name:
        block = llm.model.layers
    elif "LGAI" in llm_name:
        block = llm.transformer.h
    elif 'gemma-2' in llm_name.lower():
        block = llm.base_model.layers
    elif 'gemma-3' in llm_name.lower():
        block = llm.language_model.layers
    else:
        raise ValueError(f"Unsupported model: {llm_name}")
    return block


def get_num_layers(llm_name):
    if llm_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        return 32
    elif llm_name == "meta-llama/Meta-Llama-3.1-8B":
        return 32
    elif llm_name == "meta-llama/Llama-3.1-8B":
        return 32
    elif llm_name == "Qwen/Qwen3-4B-Instruct-2507":
        return 36
    elif llm_name == "google/gemma-3-4b-it":
        return 34
    elif llm_name == "kakaocorp/kanana-1.5-8b-instruct-2505":
        return 32
    elif llm_name == "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct":
        return 30
    elif llm_name == "google/gemma-2-2b":
        return 26
    else:
        raise ValueError(f"Unsupported model: {llm_name}")

def get_mlp_down_proj(llm_name, block):
    if 'LGAI' in llm_name:
        module = block.mlp.c_proj
    else:
        module = block.mlp.down_proj
    return module


hooks = []
hook_handles = []
blocks = get_llm_block(model, args.model_name)
assert len(blocks) == get_num_layers(args.model_name)
layer_indices = list(range(len(blocks)))  # [::4] + [len(blocks) - 1] # every 4 layers and the last layer
for layer_idx in layer_indices:
    hook = HiddenStateHook()
    module = get_mlp_down_proj(args.model_name, blocks[layer_idx])
    handle = module.register_forward_hook(hook)
    hooks.append(hook)
    hook_handles.append(handle)
    
    
from tqdm import tqdm

@torch.no_grad()
def neuron_label_corr(X_pos, X_neg, device=None):
    # Decide compute device (prefer CUDA if available)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif X_pos.numel() > 0:
            device = X_pos.device
        elif X_neg.numel() > 0:
            device = X_neg.device
        else:
            device = torch.device("cpu")

    # Handle empty tensors
    if X_pos.numel() == 0 or X_neg.numel() == 0:
        # Return zero correlation if either tensor is empty (CPU tensor for consistency)
        if X_pos.numel() > 0:
            return torch.zeros(X_pos.size(1), device="cpu")
        elif X_neg.numel() > 0:
            return torch.zeros(X_neg.size(1), device="cpu")
        else:
            return torch.empty(0, device="cpu")
    
    # Move to compute device and promote to float32 for numerical stability
    X = torch.cat([X_pos, X_neg], dim=0).to(device, non_blocking=True).float()   # [N, D]
    y = torch.cat([
        torch.ones(X_pos.size(0), device=device),
        torch.zeros(X_neg.size(0), device=device)
    ], dim=0).float()  # [N]

    # 중심화
    Xc = X - X.mean(0, keepdim=True)
    yc = y - y.mean()

    num = (Xc * yc.unsqueeze(1)).sum(0)
    denom = (Xc.square().sum(0).sqrt() * yc.square().sum().sqrt()).clamp_min(1e-12)
    r = num / denom  # (D,)
    return r.to("cpu")


all_hidden_states_at_0 = {layer: [] for layer in layer_indices}
all_hidden_states_at_1 = {layer: [] for layer in layer_indices}
all_hidden_states_at_2 = {layer: [] for layer in layer_indices}
all_hidden_states_at_3 = {layer: [] for layer in layer_indices}

all_rest_of_indices_at_0 = {layer: [] for layer in layer_indices}
all_rest_of_indices_at_1 = {layer: [] for layer in layer_indices}
all_rest_of_indices_at_2 = {layer: [] for layer in layer_indices}
all_rest_of_indices_at_3 = {layer: [] for layer in layer_indices}


if len(gathered_generations) % args.batch_size != 0:
    num_mini_batch = len(gathered_generations) // args.batch_size + 1 
else:
    num_mini_batch = len(gathered_generations) // args.batch_size

pbar = tqdm(range(num_mini_batch))
for i in pbar:
    # Check memory usage before processing
    current_memory = get_memory_usage()
    pbar.set_description(f"Processing mini-batch {i+1}/{num_mini_batch} (Memory: {current_memory:.2f}GB)")
    
    # Prepare Batch
    batch = []
    for j in range(args.batch_size):
        index = i*args.batch_size + j
        if index < len(gathered_generations):
            batch.append({'text': gathered_generations[index]})
        else:
            batch.append({'text': ''})
            
    batch = custom_collate_fn_with_end_symbol_indices(batch, tokenizer)
    for hook in hooks:
        hook.end_symbol_indices = batch['end_symbol_indices']
    
    # Forward Pass
    with torch.no_grad():
        outputs = model(batch['input_ids'].to(model.device), 
                        attention_mask=batch['attention_mask'].to(model.device))
        for hook_index, hook in enumerate(hooks):
            if hook.hidden_states_at_0.ndim == 2:
                all_hidden_states_at_0[hook_index].append(hook.hidden_states_at_0)
            if hook.hidden_states_at_1.ndim == 2:
                all_hidden_states_at_1[hook_index].append(hook.hidden_states_at_1)
            if hook.hidden_states_at_2.ndim == 2:
                all_hidden_states_at_2[hook_index].append(hook.hidden_states_at_2)
            if hook.hidden_states_at_3.ndim == 2:
                all_hidden_states_at_3[hook_index].append(hook.hidden_states_at_3)

            if hook.rest_of_indices_at_0.ndim == 2:
                all_rest_of_indices_at_0[hook_index].append(hook.rest_of_indices_at_0)
            if hook.rest_of_indices_at_1.ndim == 2:
                all_rest_of_indices_at_1[hook_index].append(hook.rest_of_indices_at_1)
            if hook.rest_of_indices_at_2.ndim == 2:
                all_rest_of_indices_at_2[hook_index].append(hook.rest_of_indices_at_2)
            if hook.rest_of_indices_at_3.ndim == 2:
                all_rest_of_indices_at_3[hook_index].append(hook.rest_of_indices_at_3)
    
    # Clear hook data to free memory
    for hook in hooks:
        hook.clear()
    
    # Periodic memory cleanup
    if i % 10 == 0:  # Every 10 batches
        cleanup_memory()
        if check_memory_limit():
            print(f"Memory limit exceeded at batch {i}. Consider reducing batch_size or random_sample_ratio.")
    
    # For Debugging
    # if i > 3:
    #     break 

# [ALL Cases x Dim]
corr_results_all_at = {0:[], 1:[], 2:[], 3:[]}
for hook_index, hook in enumerate(hooks):
    # Check if lists are not empty before concatenating
    if len(all_hidden_states_at_0[hook_index]) > 0:
        all_hidden_states_at_0[hook_index] = torch.concat(all_hidden_states_at_0[hook_index], dim=0)
    else:
        all_hidden_states_at_0[hook_index] = torch.tensor([])
        
    if len(all_hidden_states_at_1[hook_index]) > 0:
        all_hidden_states_at_1[hook_index] = torch.concat(all_hidden_states_at_1[hook_index], dim=0)
    else:
        all_hidden_states_at_1[hook_index] = torch.tensor([])
        
    if len(all_hidden_states_at_2[hook_index]) > 0:
        all_hidden_states_at_2[hook_index] = torch.concat(all_hidden_states_at_2[hook_index], dim=0)
    else:
        all_hidden_states_at_2[hook_index] = torch.tensor([])
        
    if len(all_hidden_states_at_3[hook_index]) > 0:
        all_hidden_states_at_3[hook_index] = torch.concat(all_hidden_states_at_3[hook_index], dim=0)
    else:
        all_hidden_states_at_3[hook_index] = torch.tensor([])
    
    if len(all_rest_of_indices_at_0[hook_index]) > 0:
        all_rest_of_indices_at_0[hook_index] = torch.concat(all_rest_of_indices_at_0[hook_index], dim=0)
    else:
        all_rest_of_indices_at_0[hook_index] = torch.tensor([])
        
    if len(all_rest_of_indices_at_1[hook_index]) > 0:
        all_rest_of_indices_at_1[hook_index] = torch.concat(all_rest_of_indices_at_1[hook_index], dim=0)
    else:
        all_rest_of_indices_at_1[hook_index] = torch.tensor([])
        
    if len(all_rest_of_indices_at_2[hook_index]) > 0:
        all_rest_of_indices_at_2[hook_index] = torch.concat(all_rest_of_indices_at_2[hook_index], dim=0)
    else:
        all_rest_of_indices_at_2[hook_index] = torch.tensor([])
        
    if len(all_rest_of_indices_at_3[hook_index]) > 0:
        all_rest_of_indices_at_3[hook_index] = torch.concat(all_rest_of_indices_at_3[hook_index], dim=0)
    else:
        all_rest_of_indices_at_3[hook_index] = torch.tensor([])  
    
    print(all_hidden_states_at_0[hook_index].shape, all_rest_of_indices_at_0[hook_index].shape)
    print(all_hidden_states_at_1[hook_index].shape, all_rest_of_indices_at_1[hook_index].shape)
    print(all_hidden_states_at_2[hook_index].shape, all_rest_of_indices_at_2[hook_index].shape)
    print(all_hidden_states_at_3[hook_index].shape, all_rest_of_indices_at_3[hook_index].shape)
    
    corrs_at_0 = neuron_label_corr(all_hidden_states_at_0[hook_index], all_rest_of_indices_at_0[hook_index])
    corr_results_all_at[0].append(corrs_at_0)
    
    corrs_at_1 = neuron_label_corr(all_hidden_states_at_1[hook_index], all_rest_of_indices_at_1[hook_index])
    corr_results_all_at[1].append(corrs_at_1)
    
    corrs_at_2 = neuron_label_corr(all_hidden_states_at_2[hook_index], all_rest_of_indices_at_2[hook_index])
    corr_results_all_at[2].append(corrs_at_2)
    
    corrs_at_3 = neuron_label_corr(all_hidden_states_at_3[hook_index], all_rest_of_indices_at_3[hook_index])
    corr_results_all_at[3].append(corrs_at_3)
    
import pickle 
os.makedirs(args.save_dir, exist_ok=True)
pickle.dump(corr_results_all_at, open(f'{args.save_dir}/corr_results_all_at.pkl', 'wb'))
print("Saved the corr_results_all_at to ", f'{args.save_dir}/corr_results_all_at.pkl')