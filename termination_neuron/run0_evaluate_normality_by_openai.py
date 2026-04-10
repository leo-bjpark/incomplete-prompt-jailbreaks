import os 
import json
import argparse
from tqdm import tqdm
import openai
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
parser.add_argument("--eval_model_name", type=str, default="gpt-5-mini")
parser.add_argument("--eval_task", type=str, default="run1_complete_prompting")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")

args = parser.parse_args()
args.save_dir = f"outputs/{args.eval_task}/{args.model_name}"
if not os.path.exists(args.save_dir):
    raise ValueError(f"Save directory {args.save_dir} does not exist")

# ================================================
# Initialize OpenAI client
client = OpenAI(api_key=args.openai_api_key)

# ================================================
# construct data 
data = json.load(open(os.path.join(args.save_dir, "all_outputs.json"), "r"))

def construct_data():
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
    
    data_list = []
    for i in range(len(data)):
        input_text = data[i]['input_text'].replace("\n", "  ")
        output_text = data[i]['output_text'].replace("\n", "  ")
        text = eval_prompt.format(input_text=input_text, output_text=output_text)
        data_list.append({
            'input_text': input_text,
            'output_text': output_text,
            'eval_prompt': text
        })
        
    return data_list

dataset = construct_data()
print("Total number of data : ", len(dataset))
print("--------------------------------")

# ================================================
# Process data with OpenAI API
all_outputs = []
pbar = tqdm(dataset, desc='Processing samples')

def call_openai_api(prompt, max_retries=3):
    for retry in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=args.eval_model_name,  # 예: "gpt-4o-mini"
                messages=[
                    {
                        "role": "system",
                        "content":  "Return exactly one line like: Evaluation: Case X (X in {1,2,3,4})."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=16,  # Responses API는 max_output_tokens
                # temperature=0
            )
            # text 추출
            content = resp.choices[0].message.content.strip()
            return content
        except Exception as e:
            print(f"[ERROR attempt {retry+1}/{max_retries}] {e}", flush=True)
            if retry < max_retries - 1:
                import time; time.sleep(1)
            else:
                return f"[API_ERROR]: {e}"

for idx, sample in enumerate(pbar):
    pbar.set_description("Run (1): Evaluate Logical Answers")
    
    # Call OpenAI API
    evaluation = call_openai_api(sample['eval_prompt'])
    all_outputs.append({
        'input_text': sample['input_text'],
        'output_text': sample['output_text'],
        'evaluation': evaluation
    })
    
    if idx == 0:
        pbar.set_postfix(Generated=evaluation[:15].replace("\n", ""))
    
# ================================================
# save the features 
json.dump(all_outputs, open(f'{args.save_dir}/evaluation_outputs_by_openai.json', 'w'), indent=4)
print("Saved the answers to ", f'{args.save_dir}/evaluation_outputs_by_openai.json')