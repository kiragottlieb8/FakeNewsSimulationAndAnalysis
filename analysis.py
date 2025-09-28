import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stats.csv")
#col names: network_type,noise,threshold,trust_min,trust_max,truth_fraction,mean_persistent,std_persistent

df["trust_range"] = df.apply(lambda row: f"{row['trust_min']}-{row['trust_max']}", axis=1)

# Fixed noise level for analysis
fixed_noise = 0.1

#fig 1(3 graphs separated by graph topography each showing truth fraction on the X axis,
# mean persistence on the Y axis & different lines for trust ranges)
fixed_threshold =df["threshold"].iloc[0]
df_fixed = df[(df["noise"] == fixed_noise) & (df["threshold"] == fixed_threshold)]
network_types = df_fixed["network_type"].unique()

fig, axes = plt.subplots(1, len(network_types), figsize=(6 * len(network_types), 5), sharey=True)
for ax, net in zip(axes, network_types):
    subset = df_fixed[df_fixed["network_type"] == net]
    for trust_range, group in subset.groupby("trust_range"):
        ax.plot(group["truth_fraction"], group["mean_persistent"], marker="o", label=trust_range)
        ax.fill_between(
            group["truth_fraction"],
            group["mean_persistent"] - group["std_persistent"],
            group["mean_persistent"] + group["std_persistent"],
            alpha=0.2
        )
    ax.set_title(f"{net} Network (threshold = {fixed_threshold})")
    ax.set_ylabel("Mean Persistence")
    ax.set_xlabel("Truth Fraction")
    ax.legend(title="Trust Range", fontsize=8)
plt.tight_layout()
plt.show()

#fig 2(5 graphs separated by threshold each showing truth fraction on the X axis,
# mean persistence on the Y axis & different lines for trust ranges)
df_ER = df[df["network_type"] == "ER"]
thresholds = sorted(df_ER["threshold"].unique())

fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 5), sharey=True)
for ax, thr in zip(axes, thresholds):
    subset = df_ER[(df_ER["threshold"] == thr) & (df_ER["noise"] == fixed_noise)]
    for trust_range, group in subset.groupby("trust_range"):
        ax.plot(group["truth_fraction"], group["mean_persistent"], marker="o", label=trust_range)
        ax.fill_between(
            group["truth_fraction"],
            group["mean_persistent"] - group["std_persistent"],
            group["mean_persistent"] + group["std_persistent"],
            alpha=0.2
        )
    ax.set_title(f"ER (Threshold={thr})")
    ax.set_ylabel("Mean Persistence")
    ax.set_xlabel("Truth Fraction")
    ax.legend(title="Trust Range", fontsize=8)
plt.tight_layout()
plt.show()

#fig 3(6 graphs separated by truth fraction each showing threshold on the X axis,
# mean persistence on the Y axis & different lines for trust ranges)
truth_values = sorted(df_ER["truth_fraction"].unique())

fig, axes = plt.subplots(1, len(truth_values), figsize=(6 * len(truth_values), 5), sharey=True)
for ax, truth_val in zip(axes, truth_values):
    subset = df_ER[(df_ER["truth_fraction"] == truth_val) & (df_ER["noise"] == fixed_noise)]
    for trust_range, group in subset.groupby("trust_range"):
        ax.plot(group["threshold"], group["mean_persistent"], marker="o", label=trust_range)
        ax.fill_between(
            group["threshold"],
            group["mean_persistent"] - group["std_persistent"],
            group["mean_persistent"] + group["std_persistent"],
            alpha=0.2
        )
    ax.set_title(f"ER (Truth Fraction={truth_val})")
    ax.set_ylabel("Mean Persistence")
    ax.set_xlabel("Threshold")
    ax.legend(title="Trust Range", fontsize=8)
plt.tight_layout()
plt.show()

#fig 4(4 graphs separated by trust range each showing truth fraction on the X axis,
# mean persistence on the Y axis & different lines for thresholds)
trust_ranges = sorted(df_ER["trust_range"].unique())

fig, axes = plt.subplots(1, len(trust_ranges), figsize=(6 * len(trust_ranges), 5), sharey=True)
for ax, trust_range in zip(axes, trust_ranges):
    subset = df_ER[(df_ER["trust_range"] == trust_range) & (df_ER["noise"] == fixed_noise)]
    for truth_val, group in subset.groupby("threshold"):
        ax.plot(group["truth_fraction"], group["mean_persistent"], marker="o", label=truth_val)
        ax.fill_between(
            group["truth_fraction"],
            group["mean_persistent"] - group["std_persistent"],
            group["mean_persistent"] + group["std_persistent"],
            alpha=0.2
        )
    ax.set_title(f"ER (Trust range={trust_range})")
    ax.set_ylabel("Mean Persistence")
    ax.set_xlabel("Truth fraction")
    ax.legend(title="Threshold", fontsize=8)
plt.tight_layout()
plt.show()

