import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

paths = {
    "seed1": "./10-06_results_seed10/3d27665367ffece0b81efe299f4fcea9/score_list.pkl",
    "baseline": "./baseline_results_seed/af6f9423403e8a2dd61bca2a75bc9c81/score_list.pkl"
}

# 定义颜色
colors = {"seed1": "C1", "baseline": "C2"}

# 存储每个实验的数据
results = {}

for name, path in paths.items():
    with open(path, "rb") as f:
        data = pickle.load(f)
    results[name] = {
        "epochs": [d["epoch"] for d in data],
        "step_size": [d["step_size"] for d in data],
        "train_loss": [d["train_loss"] for d in data],
        "val_loss": [d.get("val_loss", None) for d in data],
        "train_epoch_time": [d.get("train_epoch_time", 0.0) for d in data],  # ✅ 新增
    }

# ---- 绘制子图 ----
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# 子图1: step_size
for name, data in results.items():
    axes[0].plot(data["epochs"], data["step_size"],
                 color=colors[name], label=f"Step Size ({name})")
axes[0].set_ylabel("Step Size")
axes[0].legend()
axes[0].grid(True)

# 子图2: train_loss vs val_loss
max_loss = 5.0
for name, data in results.items():
    train_loss_clipped = np.clip(data["train_loss"], None, max_loss)
    val_loss_clipped = np.clip(
        [v if v is not None else np.nan for v in data["val_loss"]],
        None, max_loss
    )
    axes[1].plot(data["epochs"], train_loss_clipped,
                 color=colors[name], label=f"Train Loss ({name})")
    if any(v is not None for v in data["val_loss"]):
        axes[1].plot(data["epochs"], val_loss_clipped, "--",
                     color=colors[name], label=f"Val Loss ({name})")

axes[1].set_ylabel("Loss (≤ {:.1f})".format(max_loss))
axes[1].set_yscale("log")
axes[1].legend()
axes[1].grid(True)

# ✅ 子图3: 累计时间（累计求和）
for name, data in results.items():
    cumulative_time = np.cumsum(data["train_epoch_time"])
    axes[2].plot(data["epochs"], cumulative_time,
                 color=colors[name], label=f"Cumulative Time ({name})")

axes[2].set_ylabel("Cumulative Time (s)")
axes[2].set_xlabel("Epoch")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("baseline_training_compare.png", dpi=300)
plt.close()

print("✅ Saved plot as 2_training_compare.png")
