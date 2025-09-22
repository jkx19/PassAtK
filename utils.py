import numpy as np
from qwen_matheval.grader import math_equal
from qwen_matheval.parser import extract_answer, strip_string
from tqdm import tqdm

def count_different_answers(rm_responses):
    unique_answers = []
    for response in rm_responses:
        pred = extract_answer(response["response"], "math")
        score = response["score"]
        found = False
        for answer_info in unique_answers:
            answer = answer_info["answer"]
            if math_equal(pred, answer, timeout=True):
                answer_info["count"] += 1
                answer_info["reward_scores"].append(score)
                found = True
                break
        if not found:
            unique_answers.append({
                "answer": pred,
                "count": 1,
                "reward_scores": [score]
            })
    for answer_info in unique_answers:
        answer_info["reward_scores"] = np.mean(answer_info["reward_scores"])
        answer_info["freq"] = answer_info["count"]/len(rm_responses)
    return unique_answers
