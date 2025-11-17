# from transformers import AutoTokenizer, AutoConfig
# from vllm import SamplingParams
# from bytedverl.third_party.vllm import LLM
# from vllm import LLM
from datasets import load_dataset, Dataset
import argparse
import numpy as np

# from grader import grade_answer
from qwen_matheval.grader import math_equal
from qwen_matheval.parser import extract_answer, strip_string

from utils import count_different_answers

from tqdm import tqdm
import json
import os
import lox



def reward_function(dataset_name, response, raw_gt_answer):
    if dataset_name == "gsm8k":
        _, ground_truth = raw_gt_answer.split("####")
        ground_truth = strip_string(ground_truth, skip_unit=False)
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0

    elif dataset_name == "math500":
        ground_truth = str(raw_gt_answer)
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0

    elif dataset_name == "aime24" or dataset_name == "aime25":
        ground_truth = str(raw_gt_answer)
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0
    
    elif dataset_name == "minerva":
        ground_truth = str(raw_gt_answer)
        prediction = extract_answer(response, "math")
        return 1 if math_equal(prediction, ground_truth, timeout=True) else 0


def pessimistic_select_k(different_answer, k, threshold=0.2):
    different_answer = sorted(different_answer, key=lambda x: x["reward_scores"], reverse=True)
    k_answers = []
    for answer_info in different_answer:
        freq = answer_info["freq"]
        if freq >= threshold:
            answer = answer_info["answer"]
            k_answers.append(
                {
                    "response": "\\boxed{" + answer + "}",
                    "score": answer_info["reward_scores"]
                }
            )
        if len(k_answers) >= k:
            break
    return k_answers


def select_k_by_reward(different_answer, k):
    sorted_answers = sorted(different_answer, key=lambda x: x["reward_scores"], reverse=True)
    k_answers = []
    for i in range(min(k, len(sorted_answers))):
        answer = sorted_answers[i]["answer"]
        k_answers.append(
            {
                "response": "\\boxed{" + answer + "}",
                "score": sorted_answers[i]["reward_scores"]
            }
        )
    return k_answers


def majority_k(different_answer, k):
    sorted_answers = sorted(different_answer, key=lambda x: x["count"], reverse=True)
    k_answers = []
    for i in range(min(k, len(sorted_answers))):
        answer = sorted_answers[i]["answer"]
        k_answers.append(
            {
                "response": "\\boxed{" + answer + "}",
                "score": sorted_answers[i]["reward_scores"]
            }
        )
    return k_answers


def soft_best_of_n(different_answer, k, beta):
    all_soft_scores = []
    for answer_info in different_answer:
        freq = answer_info["freq"]
        soft_score = freq * np.exp(beta * answer_info["reward_scores"])
        all_soft_scores.append(soft_score)
    Z = sum(all_soft_scores)
    answer_w_soft_scores = []
    for answer_info in different_answer:
        freq = answer_info["freq"]
        soft_score = freq * np.exp(beta * answer_info["reward_scores"]) / Z
        answer_w_soft_scores.append({
            "answer": answer_info["answer"],
            "soft_score": soft_score,
            "freq": freq,
            "reward_scores": answer_info["reward_scores"]
        })
    probabilities = [x["soft_score"] for x in answer_w_soft_scores]
    selected_indices = np.random.choice(len(different_answer), size=k, replace=False, p=probabilities)
    k_answers = []
    for idx in selected_indices:
        answer = answer_w_soft_scores[idx]["answer"]
        k_answers.append(
            {
                "response": "\\boxed{" + answer + "}",
                "score": answer_w_soft_scores[idx]["reward_scores"]
            }
        )
    return k_answers
        
    # sorted_responses = sorted(responses_and_scores, key=lambda x: x["score"], reverse=True)
    
    # return sorted_responses[:k]



if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters for text generation.')

    # Add arguments
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate for each task")
    parser.add_argument("--real_N", type=int, default=50, help="Number of real samples to generate for each task")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500", "aime24", "minerva", "aime25"], help="Dataset to use for generation")
    parser.add_argument("--input_file", type=str, default=None, help="Input file containing questions (if not using built-in datasets)")
    parser.add_argument("--pass_at", type=int, default=3, help="Number of samples to select based on RM scores")
    parser.add_argument("--method", type=str, default="pessimistic", choices=["pessimistic", "majority", "reward", "softBoN"], help="Method to select k samples")
    parser.add_argument("--threshold", type=float, default="0.0")
    parser.add_argument('--output_file', type=str, default='', help='File to save the final results.')
    parser.add_argument("--threading", type=int, default=8, help="Number of threads for parallel processing")
    

    args = parser.parse_args()

    dataset_name = args.dataset
    input_file = args.input_file
    num_samples_per_task = args.num_samples
    k = args.pass_at
    threshold = args.threshold
    input_file = args.input_file
    output_file = args.output_file
    real_N = args.real_N
    num_threads = args.threading
    if real_N > num_samples_per_task:
        print("Error: real_N should be less than or equal to num_samples_per_task")
        exit(1)
    # method = args.method


    if dataset_name == "gsm8k":
        # dataset = load_dataset("gsm8k", "main")
        problems = Dataset.from_parquet(f"dataset/gsm8k_hard.parquet")
        gt_column_name = "answer"
        question_column_name = "question"
    elif dataset_name == "aime25":
        problems = Dataset.from_parquet(f"dataset/aime25.parquet")
        gt_column_name = "answer"
        question_column_name = "problem"
    elif dataset_name == "aime24":
        problems = Dataset.from_parquet(f"dataset/aime24.parquet")
        gt_column_name = "answer"
        question_column_name = "problem"
    elif dataset_name == "minerva":
        problems = Dataset.from_parquet(f"dataset/minerva_math.parquet")
        gt_column_name = "answer"
        question_column_name = "question"
    elif dataset_name == "math500":
        problems = Dataset.from_parquet(f"dataset/math500.parquet")
        gt_column_name = "answer"
        question_column_name = "problem"

    dataset_size = len(problems)

    f = open(input_file, "r")
    prediction_w_scores = json.load(f)
    f.close()

    results = []

    # not_all_pass = []
    different_answers = []

    if os.path.exists(output_file):
        f = open(output_file, "r")
        different_answers = json.load(f)
        f.close()
    else:
        different_answer_threads = lox.thread(num_threads)(count_different_answers)
        for task_id in range(dataset_size):
            response_and_scores = prediction_w_scores[task_id][:real_N]
            different_answer_threads.scatter(response_and_scores)
            # different_answer = count_different_answers(response_and_scores)
        different_answers = different_answer_threads.gather(tqdm=True)
    
        f = open(output_file, "w")
        json.dump(different_answers, f, indent=4)
        f.close()

    for task_id in tqdm(range(dataset_size)):
        different_answer = different_answers[task_id]
        if args.method == "pessimistic":
            response_and_scores = pessimistic_select_k(different_answer, k=k, threshold=threshold)
        elif args.method == "majority":
            response_and_scores = majority_k(different_answer, k=k)
        elif args.method == "reward":
            response_and_scores = select_k_by_reward(different_answer, k=k)
        elif args.method == "softBoN":
            beta = 1.0
            response_and_scores = soft_best_of_n(different_answer, k=k, beta=beta)
        raw_ground_truth = problems[task_id][gt_column_name]
        pass_at_k = 0
        have_failed = False
        if response_and_scores[0]["response"] == "Too many answers":
            continue
        for sample_id in range(len(response_and_scores)):
            response = response_and_scores[sample_id]["response"]
            rm_score = response_and_scores[sample_id]["score"]
            gt_score = reward_function(dataset_name, response, raw_ground_truth)
            if gt_score == 1:
                pass_at_k = 1
                break
        results.append(pass_at_k)

    print(f"Pass@{k}: {np.mean(results)}")
        


