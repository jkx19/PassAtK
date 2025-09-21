import matplotlib.pyplot as plt
import numpy as np

# X values (Inference FLOPs per question)
k = np.array([1,2,3,5,10])

# Synthetic Y values (replace with your real data)
majority = [20.00,23.33,23.33,30.00,36.67]
reward = [20.00,23.33,26.67,33.33,43.33]
pessimistic = [20.00,26.67,26.67,26.67,40.00]

dataset_name = "AIME24"
model_name = "qwen2.5-7b"


# def plot_with_band(x, y, label, color):
#     plt.plot(x, y, marker='o', label=label, color=color)
#     plt.fill_between(x, np.array(y)-err, np.array(y)+err, alpha=0.2, color=color)

plt.figure(figsize=(7,5))

plt.plot(k, majority, marker='o', label='Mojority Voting', color='#1f77b4')
plt.plot(k, reward, marker='*', label='BoN', color='#ff7f0e')
plt.plot(k, pessimistic, marker='^', label='BoM', color='#2ca02c')

# Plot each method
# plot_with_band(sse_34B, "REBASE (34B)", "#fb9a99")

# Formatting
# plt.xscale("log", base=2)
plt.xlabel("Number of Candidates (k)")
plt.ylabel("Test Accuracy (%) on " + dataset_name)
# plt.title("Inference scaling (Weighted Majority)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/{model_name}_{dataset_name}.pdf", dpi=300)