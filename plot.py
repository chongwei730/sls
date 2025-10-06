import pickle
import matplotlib.pyplot as plt

paths = {
    # "momentum_4096": "./momentum_0.5_4096_results/9ca41c392e07ce8dcf3e71a887c4a6f7/score_list.pkl",
    "momentum_1024": "./momentum_beta0.5_results/3f43a30df6128db1856771ad5331f667/score_list.pkl",
    "results": "./results/35f255aa0f583b82cee9b5d25a38d58a/score_list.pkl"

}

# 定义颜色列表
colors = { "momentum_1024": "C1", "results": "C2"}

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
        "train_acc": [d.get("train_acc", None) for d in data],
        "val_acc": [d.get("val_acc", None) for d in data],
    }
print(results)

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
for name, data in results.items():
    axes[1].plot(data["epochs"], data["train_loss"], 
                 color=colors[name], label=f"Train Loss ({name})")
    if any(v is not None for v in data["val_loss"]):
        axes[1].plot(data["epochs"], data["val_loss"], "--", 
                     color=colors[name], label=f"Val Loss ({name})")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

# 子图3: train_acc vs val_acc
for name, data in results.items():
    if any(v is not None for v in data["train_acc"]):
        axes[2].plot(data["epochs"], data["train_acc"], 
                     color=colors[name], label=f"Train Acc ({name})")
    if any(v is not None for v in data["val_acc"]):
        axes[2].plot(data["epochs"], data["val_acc"], "--", 
                     color=colors[name], label=f"Val Acc ({name})")
axes[2].set_ylabel("Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("1_training_compare.png", dpi=300)
plt.close()
