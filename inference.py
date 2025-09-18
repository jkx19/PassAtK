from transformers import AutoTokenizer, AutoConfig
from vllm import SamplingParams
# from bytedverl.third_party.vllm import LLM
from vllm import LLM
from datasets import load_dataset, Dataset
import os
import gzip
import argparse
import numpy as np

# from grader import grade_answer
from qwen_matheval.grader import math_equal
from qwen_matheval.parser import extract_answer, strip_string
from typing import Iterable, Dict

from tqdm import tqdm
import json


max_model_len, tp_size = 4096, 4

import re

def reward_function(dataset_name, response, raw_gt_answer):
    if dataset_name == "gsm8k":
        _, ground_truth = raw_gt_answer.split("####")
        ground_truth = strip_string(ground_truth, skip_unit=False)
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0

    elif dataset_name == "math":
        ground_truth = extract_answer(raw_gt_answer, "math")
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0

    elif dataset_name == "aime24":
        ground_truth = str(raw_gt_answer)
        # ground_truth = extract_answer(raw_gt_answer, "math")
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters for text generation.')

    # Add arguments
    parser.add_argument("--strategy", type=str, choices=["greedy", "sample"])
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate for each task")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct", help="Model name")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "aime24"], help="Dataset to use for generation")
    parser.add_argument("--output_file", type=str, default="output.json", help="File to save the generated outputs")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for the model")

    # Parse the arguments
    args = parser.parse_args()

    dataset_name = args.dataset
    output_file = args.output_file

    if dataset_name == "gsm8k":
        # dataset = load_dataset("gsm8k", "main")
        problems = Dataset.from_parquet(f"dataset/gsm8k_hard.parquet")
        gt_column_name = "answer"
        question_column_name = "question"
    elif dataset_name == "math":
        # dataset = load_dataset("lighteval/MATH", "all")
        problems = Dataset.from_parquet(f"dataset/math_test.parquet")
        gt_column_name = "solution"
        question_column_name = "problem"
    elif dataset_name == "aime24":
        problems = Dataset.from_parquet(f"dataset/aime24.parquet")
        gt_column_name = "Answer"
        question_column_name = "Problem"


    # problems = problems.select(range(10))
    dataset_size = len(problems)
    num_samples_per_task = args.num_samples
    strategy = args.strategy
    model_name = args.model_name
    system_prompt = args.system_prompt
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    if strategy == "greedy":
        sampling_params = SamplingParams(n=num_samples_per_task, top_k=-1,temperature=0, max_tokens=8192)

    elif strategy == "sample":
        sampling_params = SamplingParams(n=num_samples_per_task, top_k=16, temperature=0.7, top_p=0.9, min_p=0.01, max_tokens=8192)

    all_prompts = []
    cnt = 0
    for task_id in range(dataset_size):
        if dataset_name == "gsm8k":
            cur_message = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": problems[task_id][question_column_name]}
            ]
        elif dataset_name == "math":
            cur_message = [
                {"role": "user", "content": problems[task_id][question_column_name]}
            ]
        elif dataset_name == "aime24":
            cur_message = [{"role": "user", "content": problems[task_id][question_column_name]}]
        if system_prompt is not None:
            cur_message = [{"role": "system", "content": system_prompt}] + cur_message
        
        all_prompts.append(cur_message)


    llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True, dtype="bfloat16")

    outputs = llm.chat(all_prompts, sampling_params=sampling_params)
    sample_cnt = 0
    samples = []

    samples = {task_id: [] for task_id in range(dataset_size)}

    for task_id in range(dataset_size):
        for sample_id in range(num_samples_per_task):
            complete_result = outputs[task_id].outputs[sample_id].text
            samples[task_id].append(complete_result)

    # path = 
    f = open(output_file, "w")
    json.dump(samples, f)

    # path = f"output/{method}_{dataset_name}_{strategy}.json"
    # samples = json.load(open(path))

    # pass_rate_list = []
    # for n in range(num_samples_per_task):
    #     right, wrong = 0, 0
    #     for task_id in tqdm(range(dataset_size)):
    #         raw_ground_truth = problems[task_id][gt_column_name]
    #         raw_response = samples[task_id][n]
    #         reward = reward_function(dataset_name, raw_response, raw_ground_truth)
    #         if reward == 1:
    #             right += 1
    #         else:
    #             wrong += 1
    #     pass_rate = right / (right + wrong)
    #     pass_rate_list.append(pass_rate)
    # pass_rate_list = np.array(pass_rate_list)
    # mean = np.mean(pass_rate_list)
    # stdv = np.std(pass_rate_list)

    # print(f"Method: {args.method}, Strategy: {strategy}, Mean: {mean}, Stdv: {stdv}")

    # path = f"output/{method}_{dataset_name}_{strategy}.json"
    # f = open(path, "w")
    # json.dump(samples, f)

    