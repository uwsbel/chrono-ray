from pathlib import Path
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# COLORS
# -----------------------------
LILAC = "#D9B3F2"
BABY_LILAC = "#DFBAF7"
DARK_LILAC = "#8959A8"

# -----------------------------
# INPUT FILE
# -----------------------------
txt_path = Path("./chr_ray_log_20260511_004937.txt")

# Number of datapoints used in least squares
N_DATA = 50

# -----------------------------
# READ LOG FILE
# -----------------------------
text = txt_path.read_text()

# Pull values from txt file
objectives = [
    float(x)
    for x in re.findall(r"objective:\s*([0-9.eE+-]+)", text)
]

times = [
    float(x)
    for x in re.findall(r"time_total_s:\s*([0-9.eE+-]+)", text)
]

# -----------------------------
# BUILD DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "trial": range(1, len(objectives) + 1),
    "objective_sse": objectives,
    "rmse": [math.sqrt(obj / N_DATA) for obj in objectives],
    "time_total_s": times[:len(objectives)]
})

df["best_rmse_so_far"] = df["rmse"].cummin()
df["cumulative_time_s"] = df["time_total_s"].cumsum()

# Save parsed values
df.to_csv("training_results_parsed.csv", index=False)

# -----------------------------
# FIGURE 1: RMSE OVER TRIALS
# -----------------------------
plt.figure(figsize=(7, 4.5))

plt.plot(
    df["trial"],
    df["rmse"],
    "o",
    color=BABY_LILAC,
    markersize=7,
    label="Trial RMSE"
)

plt.plot(
    df["trial"],
    df["best_rmse_so_far"],
    "o-",
    color=DARK_LILAC,
    linewidth=2,
    markersize=5,
    label="Best RMSE so far"
)

plt.yscale("log")

plt.xlabel("Trial")
plt.ylabel("RMSE")
plt.title("Training RMSE over trials")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "rmse_over_trials.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# FIGURE 2: RUNTIME PER TRIAL
# -----------------------------
plt.figure(figsize=(7, 4.5))

plt.bar(
    df["trial"],
    df["time_total_s"],
    color=LILAC,
    edgecolor=DARK_LILAC
)

plt.xlabel("Trial")
plt.ylabel("Runtime (s)")
plt.title("Runtime per trial")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "runtime_per_trial.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# FIGURE 3: BEST RMSE VS TIME
# -----------------------------
plt.figure(figsize=(7, 4.5))

plt.plot(
    df["cumulative_time_s"],
    df["best_rmse_so_far"],
    "o-",
    color=DARK_LILAC,
    linewidth=2,
    markersize=5
)

plt.yscale("log")

plt.xlabel("Cumulative training time (s)")
plt.ylabel("Best RMSE so far")

plt.title("Best RMSE vs training time")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "best_rmse_vs_training_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# PRINT SUMMARY
# -----------------------------
print(df)

print(f"\nBest RMSE: {df['rmse'].min():.4f}")
print(f"Best trial: {df.loc[df['rmse'].idxmin(), 'trial']}")