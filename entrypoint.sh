pip3 install vllm
pip3 install datasets
python3 -m pip install pylatexenc
pip install accelerate
cd latex2sympy
pip install -e .
cd ..
pip3 install word2number
mkdir output
pip3 install matplotlib
pip3 install lox



# hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/user/kaixuanji/dqo/gln/exp/models/gemma-1.1-7b-MATH_ppo_kl001/global_step_100 ./checkpoints && \
# python3 qwen_math.py --top_p=0.9 --top_k=16 --threshold=0.01 --split test --model_name "checkpoints/global_step_100" --num_samples 1 --method "ppo-s" --dataset "math" && \
# python3 qwen_math.py --top_k=1 --threshold=0.01 --split test --model_name "checkpoints/global_step_100" --num_samples 1 --method "ppo-g" --dataset "math" && \
# python3 performance.py --dataset_name math --method "ppo-s" && \
# python3 performance.py --dataset_name math --method "ppo-s"
# python3 evaluation.py --model_name checkpoints/hf_ckpt --dataset aime24 --method base --num_samples 1 --strategy greedy