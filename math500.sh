export HF_HOME=/data/huggingface
# conda activate vllm
export CUDA_VISIBLE_DEVICES=4,5,6,7

model_name="Qwen/Qwen2.5-Math-1.5B-Instruct"
dataset="math500"
reward_model_name="nvidia/AceMath-7B-RM"
N=500
K=10
output_file="output/qwen25_15_${dataset}_${N}.json"
prediction_w_rm_score_file="output/qwen25_15_${dataset}_${N}_with_scores.json"

python3 inference.py --model_name ${model_name} --dataset ${dataset} --output_file ${output_file} --num_samples ${N} --strategy sample
python3 reward.py --model_name ${reward_model_name} --dataset ${dataset} --num_samples_per_task ${N}
python3 grader.py --dataset ${dataset} --input_file ${prediction_w_rm_score_file} --pass_at ${K} --num_samples ${N} --method majority
# python3 grader.py --dataset ${dataset} --input_file ${prediction_w_rm_score_file} --pass_at ${K} --num_samples ${N} --method reward
# python3 grader.py --dataset ${dataset} --input_file ${prediction_w_rm_score_file} --pass_at ${K} --num_samples ${N} --method pessimistic --threshold 0.005
