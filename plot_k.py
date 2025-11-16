import matplotlib.pyplot as plt
import numpy as np


def plot_figure(k, majority, reward, pessimistic, dataset_name, model_name):
    plt.figure(figsize=(7,5))

    plt.rcParams['font.size'] = 16

    plt.plot(k, majority, marker='o', label='Majority Voting', color='#1f77b4')
    plt.plot(k, reward, marker='*', label='BoN', color='#ff7f0e')
    plt.plot(k, pessimistic, marker='^', label='BoM', color='#2ca02c')

    # plt.xscale("log", base=2)
    plt.xlabel("Number of Candidates (k)")
    plt.ylabel("Test Accuracy (%) on " + dataset_name)
    # plt.title("Inference scaling (Weighted Majority)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_{dataset_name}.pdf", dpi=300)



k = np.array([1,2,3,5,10])


majority = [86.4,87.8,88.2,88.6,89.6]
reward = [83.2,86.8,88.4,88.8,89.6]
pessimistic = [86.4,88.8, 89.2,89.6, 89.8]
dataset_name = "MATH-500"
model_name = "qwen3-4b"
plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)


majority = [82.7, 89.4, 90.9, 91.8, 92.7]
reward = [84.5, 90.9, 92.4, 92.4, 92.7]
pessimistic = [86.5, 91.5, 91.8, 92.1, 92.1]
dataset_name = "GSM8K"
model_name = "qwen3-4b"
plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)


majority = [36.7, 43.3, 46.7, 46.7, 60.0]
reward = [53.3, 60.0, 63.3, 70.0, 73.3]
pessimistic = [53.3, 66.7, 70.0, 70.0, 70.0]
dataset_name = "AIME24"  # Replace with the actual dataset name
model_name = "qwen3-4b"      # Replace with the actual model name
plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)


majority = [61.00, 81.52, 86.80, 90.32, 93.26]
reward = [82.11, 88.27, 91.50, 92.36, 93.26]
pessimistic = [81.82, 88.86, 90.92, 91.79, 92.67]
dataset_name = "GSM8K"  # Replace with the actual dataset name
model_name = "qwen2.5-1.5b"      # Replace with the actual model name
plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)


majority = [23.33, 26.67, 26.67, 26.67, 33.33]
reward = [20.00, 23.33, 23.33, 23.33, 30.00]
pessimistic = [20.00, 23.33, 26.67, 33.33, 33.33]
dataset_name = "AIME24"  # Replace with the actual dataset name
model_name = "qwen2.5-1.5b"      # Replace with the actual model name
plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)



majority = [75.2, 81.4, 83.0, 85.8, 88.4]
reward = [77.2, 80.8, 83.4, 85.8, 87.4]
pessimistic = [79.3, 82.6, 84.4, 86.3, 87.6]
dataset_name = "MATH-500"  # Replace with the actual dataset name
model_name = "qwen2.5-1.5b"      # Replace with the actual model name
plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)


# Data points for the new results
# majority = [88.56, 92.38, 93.55, 94.43, 95.01]
# reward = [87.97, 93.26, 93.84, 94.72, 94.72]
# pessimistic = [88.86, 92.96, 93.26, 93.84, 94.13]
# dataset_name = "GSM8K"  # Replace with the actual dataset name
# model_name = "qwen2.5-7b"      # Replace with the actual model name
# plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)


# majority_new = [83.8, 85.2, 87.6, 88.6, 89.8]
# reward_new = [79.4, 83.2, 84.6, 86.2, 87.8]
# pessimistic_new = [81.6, 84.0, 85.0, 86.6, 89.2]
# dataset_name_new = "MATH-500"  # Replace with the actual dataset name
# model_name_new = "qwen2.5-7b"     # Replace with the actual model name
# plot_figure(k, majority_new, reward_new, pessimistic_new, dataset_name_new, model_name_new)


# majority = [23.33, 26.67, 26.67, 33.33, 36.67]
# reward = [20.00, 23.33, 23.33, 30.00, 36.67]
# pessimistic = [20.00, 23.33, 26.67, 33.33, 43.33]
# dataset_name = "AIME24"  # Replace with the actual dataset name
# model_name = "qwen2.5-7b"  # Replace with the actual model name
# plot_figure(k, majority, reward, pessimistic, dataset_name, model_name)



# def plot_with_band(x, y, label, color):
#     plt.plot(x, y, marker='o', label=label, color=color)
#     plt.fill_between(x, np.array(y)-err, np.array(y)+err, alpha=0.2, color=color)

