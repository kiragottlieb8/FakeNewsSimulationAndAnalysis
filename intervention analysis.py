import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#loads a data file from the output of the simulation
df = pd.read_csv("stats_interventions_centralityBugFixAll1.csv")

#created a new row that consolidates trust min and max to create a range categorical variable
df["trust_range"] = df["trust_min"].astype(str) + "–" + df["trust_max"].astype(str)

# finds all differnt types of network topologies, noises and thresholds
network_types = df["network_type"].unique()
noises = df["noise"].unique()
thresholds = df["threshold"].unique()

#this established a directory to save the graphs to
output_dir = "intervention_heatmaps"
os.makedirs(output_dir, exist_ok=True)

# loops through all combinations of network topology, noise and threshold
for net in network_types:
    for noise in noises:
        for thr in thresholds:
            #takes data that only matches the given conbination
            subset = df[
                (df["network_type"] == net) &
                (df["noise"] == noise) &
                (df["threshold"] == thr)
            ]

            if subset.empty:
                continue

            truth_vals = np.sort(subset["truth_fraction"].unique())
            trust_ranges = subset["trust_range"].unique()

            # Initialize matrices 
            betweenness_diff = np.zeros((len(truth_vals), len(trust_ranges)))
            degree_diff = np.zeros((len(truth_vals), len(trust_ranges)))
            top_minus_degree = np.zeros((len(truth_vals), len(trust_ranges)))

            # Fill matrices with the final persistence (normalized) in each combo of truth vals(row) and trust ranges(col). 
            for i, t in enumerate(truth_vals):
                for j, tr in enumerate(trust_ranges):
                    row = subset[(subset["truth_fraction"] == t) & (subset["trust_range"] == tr)].iloc[0]
                    max_false = 1 - t
                    betweenness_diff[i, j] = (row["betweenness_100"] - row["baseline_100"]) / max_false
                    degree_diff[i, j] = (row["degree_100"] - row["baseline_100"]) / max_false
                    top_minus_degree[i, j] = (row["betweenness_100"] - row["degree_100"]) / max_false

            max_abs = max(
                np.abs(betweenness_diff).max(),
                np.abs(degree_diff).max(),
                np.abs(top_minus_degree).max()
            )

            # sets up plot for 3 heatmaps  
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            heatmaps = [betweenness_diff, degree_diff, top_minus_degree]
            titles = ["Top betweenness − Baseline", "Top degree − Baseline", "Betweenness − Degree"]

            for ax, mat, title in zip(axes, heatmaps, titles):
                im = ax.imshow(mat, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
                ax.set_xticks(range(len(trust_ranges)))
                ax.set_xticklabels(trust_ranges)
                ax.set_yticks(range(len(truth_vals)))
                ax.set_yticklabels(truth_vals)
                ax.set_xlabel("Trust range")
                ax.set_title(title)

                # Annotate values
                for l in range(mat.shape[0]):
                    for m in range(mat.shape[1]):
                        ax.text(m, l, f"{mat[l, m]:.2f}", ha="center", va="center", fontsize=9)

            axes[0].set_ylabel("Initial truth fraction")

            # label for the entire plot (specifies the constants - aka the combination that is being shown(the triple nested for loop values))
            fig.suptitle(
                f"Network={net}, Noise={noise}, Threshold={thr}\n"
                "Normalized difference in persistent false beliefs",
                fontsize=14
            )

            cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized difference in # false beliefs persistent \n(Red = more persistent(worsening)\nBlue = fewer persistent(improvement))", fontsize=10)
            #plt.show()

            #saves each figure to a png
            filename = f"{net}_noise{noise}_thr{thr}.png".replace(".", "p")
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close(fig)  # Close to save memory
