export HF_HOME=/data/huggingface
# conda activate vllm
# export CUDA_VISIBLE_DEVICES=4,5,6,7

model_name="Qwen/Qwen2.5-Math-1.5B-Instruct"
dataset="aime24"
reward_model_name="nvidia/AceMath-7B-RM"
Sample_N=1000
Real_N=1000
K=40
num_threads=10
inference_output_file="output/qwen2.5_1.5b_${dataset}_${Sample_N}.json"
prediction_w_rm_score_file="output/qwen2.5_1.5b_${dataset}_${Sample_N}_with_scores.json"
different_answers_file="output/qwen2.5_1.5b_${dataset}_${Sample_N}_${Real_N}_different_answers.json"

python3 inference.py --model_name ${model_name} --dataset ${dataset} --output_file ${inference_output_file} --num_samples ${Sample_N} --strategy sample
python3 reward.py --model_name ${reward_model_name} --dataset ${dataset} --num_samples_per_task ${Sample_N} --input_file ${inference_output_file} --output_file ${prediction_w_rm_score_file}
python3 grader.py --dataset ${dataset} --input_file ${prediction_w_rm_score_file} --pass_at ${K} --num_samples ${Sample_N} --real_N ${Real_N} --threading ${num_threads} --output_file ${different_answers_file} --method majority
python3 grader.py --dataset ${dataset} --input_file ${prediction_w_rm_score_file} --pass_at ${K} --num_samples ${N} --method reward --output_file ${different_answers_file}
