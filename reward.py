import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import os
import argparse
from datasets import load_dataset, Dataset
import json
from tqdm import tqdm

# model_name = "nvidia/AceMath-7B-RM" # Path to the model
device = "auto" # the device to load the model onto

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a reward model and perform inference.')
    parser.add_argument('--model_name', type=str, required=True, help='The name or path of the model to load.')
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "aime24"], help="Dataset to use for generation")
    parser.add_argument("--num_samples_per_task", type=int, default=5, help="Number of samples to score for each task")
    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset
    num_samples_per_task = args.num_samples_per_task


    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        device_map=device, 
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if dataset_name == "gsm8k":
        # dataset = load_dataset("gsm8k", "main")
        problems = Dataset.from_parquet(f"dataset/gsm8k_hard.parquet")
        gt_column_name = "answer"
        question_column_name = "question"

    dataset_size = len(problems)

    f = open(f"output/qwen25_gsm8k_{num_samples_per_task}.json", "r")
    predictions = json.load(f)
    f.close()

    pred_with_scores = [[] for _ in range(dataset_size)]

    pbar = tqdm(total=dataset_size * num_samples_per_task, desc="Scoring samples")

    for task_id in range(dataset_size):
        question = problems[task_id][question_column_name]
        for sample_id in range(num_samples_per_task):
            response = predictions[str(task_id)][sample_id]
            conversation = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]
            conversation_str = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )

            input_ids = tokenizer.encode(
                conversation_str, 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(model.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                score = outputs[0][0].item()

            pred_with_scores[task_id].append({
                "response": response,
                "score": score
            })
            pbar.update(1)

    output_file = f"output/qwen25_gsm8k_{num_samples_per_task}_with_scores.json"
    with open(output_file, "w") as f:
        json.dump(pred_with_scores, f)  
    f.close()
            
            


# prompt = """Compute
# \[\sum_{n = 1}^\infty \frac{F_{n + 1}}{F_n F_{n + 2}},\]where $F_n$ denotes the $n$th Fibonacci number, so $F_0 = 0$ and $F_1 = 1.$"""

# resp_1 = """To compute \[\sum_{n=1}^\infty \frac{F_{n+1}}{F_n F_{n+2}},\] where \( F_n \) is the \(n\)-th Fibonacci number, we will use the recurrence relation \( F_{n+2} = F_{n+1} + F_n \) and properties of Fibonacci numbers.\n\n---\n\n### Step 1: Simplify the General Term\n\nThe given term is\n\[\frac{F_{n+1}}{F_n F_{n+2}}.\]\n\nUsing the Fibonacci recurrence relation \( F_{n+2} = F_{n+1} + F_n \), we rewrite \( \frac{1}{F_{n+2}} \) as\n\[\frac{1}{F_{n+2}} = \frac{1}{F_{n+1} + F_n}.\]\n\nThus, the term becomes\n\[\frac{F_{n+1}}{F_n F_{n+2}} = \frac{F_{n+1}}{F_n (F_{n+1} + F_n)}.\]\n\nNow, separate the fraction into partial fractions:\n\[\frac{F_{n+1}}{F_n (F_{n+1} + F_n)} = \frac{1}{F_n} - \frac{1}{F_{n+1}}.\]\n\n---\n\n### Step 2: Write the Series\n\nThe series becomes\n\[\sum_{n=1}^\infty \frac{F_{n+1}}{F_n F_{n+2}} = \sum_{n=1}^\infty \left( \frac{1}{F_n} - \frac{1}{F_{n+1}} \right).\]\n\nThis is a **telescoping series**, where most terms cancel out. Writing the first few terms explicitly:\n\[\sum_{n=1}^\infty \left( \frac{1}{F_n} - \frac{1}{F_{n+1}} \right) = \left( \frac{1}{F_1} - \frac{1}{F_2} \right) + \left( \frac{1}{F_2} - \frac{1}{F_3} \right) + \left( \frac{1}{F_3} - \frac{1}{F_4} \right) + \cdots.\]\n\nAll intermediate terms cancel, leaving:\n\[\frac{1}{F_1} - \lim_{n \to \infty} \frac{1}{F_{n+1}}.\]\n\n---\n\n### Step 3: Evaluate the Remaining Terms\n\n- The first Fibonacci number \( F_1 = 1 \), so \( \frac{1}{F_1} = 1 \).\n- As \( n \to \infty \), \( F_{n+1} \to \infty \), so \( \frac{1}{F_{n+1}} \to 0 \).\n\nThus, the sum is:\n\[\sum_{n=1}^\infty \frac{F_{n+1}}{F_n F_{n+2}} = 1 - 0 = 1.\]\n\n---\n\n### Final Answer:\n\n\[\boxed{1}\]"""

# chat = [
#     {"role": "system", "content": "Please reason step by step, and check your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt},
#     {"role": "assistant", "content": resp_1}
# ]


# conversation_str = tokenizer.apply_chat_template(
#     chat, 
#     tokenize=False, 
#     add_generation_prompt=False
# )

# input_ids = tokenizer.encode(
#     conversation_str, 
#     return_tensors="pt", 
#     add_special_tokens=False
# ).to(model.device)

# outputs = model(input_ids=input_ids)
# print(outputs[0][0]) 