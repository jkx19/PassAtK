import matplotlib.pyplot as plt
import numpy as np


def plot_figure(n, majority, reward, pessimistic, dataset_name, model_name):
    plt.figure(figsize=(7,5))

    plt.rcParams['font.size'] = 16

    plt.plot(n, majority, marker='o', label='Majority Voting', color='#1f77b4')
    plt.plot(n, reward, marker='*', label='BoN', color='#ff7f0e')
    plt.plot(n, pessimistic, marker='^', label='BoM', color='#2ca02c')

    # plt.xscale("log", base=2)
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Test Accuracy (%) on AIME24")
    # plt.title("Inference scaling (Weighted Majority)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{model_name}_pass_at_{pass_at}.pdf", dpi=300)

# X values (Inference FLOPs per question)
n = np.array([100,200,500,1000,2000])

# Extracted values from the LaTeX table
majority = [36.7, 36.7, 36.7, 36.7, 36.7]
reward = [56.7, 53.3, 53.3, 43.3, 40.0]
pessimistic = [50.0, 60.0, 53.3, 53.3, 56.7]
pass_at = 1  
model_name = "qwen3-4b"      # Replace with the actual model name
plot_figure(n, majority, reward, pessimistic, "AIME24", model_name)

# Extracted values from the LaTeX table
majority = [46.7, 46.7, 46.7, 46.7, 46.7]
reward = [66.7, 70.0, 63.3, 60.0, 63.3]
pessimistic = [56.7, 70.0, 70.0, 70.0, 70.0]
pass_at = 3  
model_name = "qwen3-4b"  # Replace with the actual model name
plot_figure(n, majority, reward, pessimistic, "AIME24", model_name)

# Updated values from the LaTeX table
majority = [50, 50, 46.7, 46.7, 46.7]
reward = [70, 70, 70, 66.7, 66.7]
pessimistic = [56.7, 70, 70, 70, 70]
pass_at = 5
model_name = "qwen3-4b"  # Replace with the actual model name
plot_figure(n, majority, reward, pessimistic, "AIME24", model_name)

# def plot_with_band(x, y, label, color):
#     plt.plot(x, y, marker='o', label=label, color=color)
#     plt.fill_between(x, np.array(y)-err, np.array(y)+err, alpha=0.2, color=color)

