import os 
import json
import torch
import datasets
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import COMMON_PROMPT

# GPU 메모리 충분하므로 캐시만 정리하고 컴파일은 유지
import torch._dynamo

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
parser.add_argument("--eval_model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
parser.add_argument("--eval_task", type=str, default="run1_complete_prompting")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=4)

args = parser.parse_args()
args.save_dir = f"outputs/{args.eval_task}/{args.model_name}"
if not os.path.exists(args.save_dir):
    raise ValueError(f"Save directory {args.save_dir} does not exist")

# ================================================
# construct data 
tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name, use_fast=True, trust_remote_code=True)
data = json.load(open(os.path.join(args.save_dir, "all_outputs.json"), "r"))
def constrct_data():
    data_dict = {
        'text' : [],
    }
    eval_prompt = (
    "You are an evaluator model. Judge the overall quality of the Generated Text on a single 5-level scale that combines "
    "(1) NATURALNESS (content relevance, coherence, completeness, usefulness) and "
    "(2) GRAMMAR/FLUENCY (grammar, phrasing, readability).\n\n"

    "Input Prompt:\n{input_text}\n\n"
    "Generated Text:\n{output_text}\n\n"

    "Classify into exactly one of the following cases:\n"
    "Case 1. Very Poor: Off-topic, incoherent, and/or major grammar issues make it hard to understand.\n"
    "Case 2. Poor: Some relevance but large gaps; many grammar issues reduce clarity.\n"
    "Case 3. Fair: Generally relevant and understandable, but uneven or with noticeable grammar/style issues.\n"
    "Case 4. Good: Clear, relevant, mostly complete, only minor grammar/style issues.\n"
    "Case 5. Excellent: Highly clear, coherent, specific, useful, and error-free or near-native fluency.\n\n"

    "Return only in the format:\n"
    "Evaluation: Case"
    )

    for i in range(len(data)):
        input_text = data[i]['input_text'].replace("\n", "  ")
        output_text = data[i]['output_text'].replace("\n", "  ")
        text = eval_prompt.format(input_text=input_text, output_text=output_text)
        text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
        data_dict['text'].append(text)
        
    dataset = datasets.Dataset.from_dict(data_dict)
    return dataset     

def custom_collate_fn(batch, tokenizer):
    texts = [x['text'] for x in batch]
    tokenized = tokenizer(texts, 
                          padding=True,            # longest in batch
                          truncation=False,        # preserve all information
                          return_tensors="pt", 
                          return_attention_mask=True,
                          )
    return tokenized    

dataset = constrct_data()
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=lambda x: custom_collate_fn(x, tokenizer)
)
print("Total number of data : ", len(dataset))
print("--------------------------------")

# ================================================
# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(args.eval_model_name, 
                                             device_map='auto',
                                            torch_dtype=torch.bfloat16,
                                            trust_remote_code=True)

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

MAX_NUM_TOKENS = 5 
all_outputs = []
pbar = tqdm(dataloader, total=len(dataloader), desc='Processing batches')

for batch_idx, batch in enumerate(pbar):
    pbar.set_description("Run (1): Evaluate Logical Answers")
    input_ids = batch["input_ids"].to(model.device)
    number_of_average_tokens = input_ids.shape[1]
    
    # Try to generate with error handling and cache clearing
    max_retries = 4
    for retry in range(max_retries):
        try:
            with torch.no_grad():   
                outputs = model.generate(input_ids, max_new_tokens=MAX_NUM_TOKENS, do_sample=False)
            break  # 성공하면 루프 탈출
        except Exception as e:
            print(f"Generation failed (attempt {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:  # 마지막 시도가 아니면
                # 캐시 정리 후 재시도
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Cache cleared, retrying...")
            else:
                print("Max retries reached, skipping this batch")
                continue  # 이 배치를 건너뛰고 다음 배치로
    
    for i in range(len(outputs)):
        all_outputs.append({
            'input_text': tokenizer.decode(input_ids[i], skip_special_tokens=True),
            'output_text': tokenizer.decode(outputs[i, input_ids.shape[1]:], skip_special_tokens=True),
        })
        if i==0:
            pbar.set_postfix(Generated=all_outputs[-1]['output_text'][:15].replace("\n", ""), 
                             Tokens=number_of_average_tokens)
    
    
# ================================================
# save the features 
json.dump(all_outputs, open(f'{args.save_dir}/normality_evaluation_outputs.json', 'w'), indent=4)
print("Saved the answers to ", f'{args.save_dir}/normality_evaluation_outputs.json')