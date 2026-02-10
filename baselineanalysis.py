import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

#loads a data file from the output of the simulation
df = pd.read_csv("stats_interventions_centralityBugFixAll1.csv")

#created a new row that consolidates trust min and max to create a range categorical variable
df["trust_range"] = df.apply(lambda row: f"{row['trust_min']}-{row['trust_max']}", axis=1)

#all analysis was done with fixed noise 0.05 but this can be changed to 0.2 (or other values 
# depending on the simulation) as well 
fixed_noise = 0.05

#this established a directory to save the graphs to
output_dir = "baseline_graphs"
os.makedirs(output_dir, exist_ok=True)

"""
function used to generalize repeated plot/graph creation code

groups = a list of data for each subplot in the format:
    [(first subplot_var value, df for that subplot_var value),... (last subplot_var value, df for that subplot_var value)]
x = the variable on the x axis
title_fn = function that shoudl be used for naming each subplot (usually just an f string that has a 
small decription and the current value of the variable for that subplot)
legend = the title of the legend (the variable that is shown through the various lines on a graph)
filename = the name of the file where you want to save that plot
"""

def plot_maker(groups, x, title_fn, legend_title, legend, filename):
    K = len(groups)
    fig, axes = plt.subplots(2, K, figsize=(6*K, 10), sharey=False)

    if K == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    #the following is for creating the un-normalized graphs
    for col, (label, subset) in enumerate(groups):
        ax = axes[0, col]
        ax.set_ylim(-1, 100)


        for line_label, group in subset.groupby(legend):
            ax.plot(group[x], group["baseline_100"], marker="o", label=line_label)
            ax.fill_between(
                group[x],
                group["baseline_100"] - group["baseline_100_std"],
                group["baseline_100"] + group["baseline_100_std"],
                alpha=0.2
            )


        #creates lines to show the max persistence when data is un-normalized
        if x == "truth_fraction":
            xmin, xmax = subset[x].min(), subset[x].max()
            xx = np.linspace(xmin, xmax, 200)
            yy = 100 * (1 - xx)
            ax.plot(xx, yy, color="black", linestyle="--", linewidth=2)
        elif x=="threshold":
            ax.set_ylim(0, 100)
            ax.axhline(y = (1-group["truth_fraction"].iloc[0])*100 , color="black", linestyle="--", linewidth=2)


        ax.set_title(title_fn(label))
        ax.set_ylabel("# persistent")
        ax.set_xlabel(x.replace("_", " ").title())

        if col == K - 1:
            ax.legend(title=legend_title, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    #the following is for creating the normalized graphs
    for col, (label, subset) in enumerate(groups):
        ax = axes[1, col]
        ax.set_ylim(-1, 102)

        #calculates normalized values
        subset = subset.copy()
        subset["max_false"] = 100 * (1 - subset["truth_fraction"])
        subset["norm"] = subset["baseline_100"]*100 / subset["max_false"].replace(0, np.nan)

        for line_label, group in subset.groupby(legend):
            ax.plot(group[x], group["norm"], marker="o", label=line_label)
            ax.fill_between(
                group[x],
                (group["baseline_100"] - group["baseline_100_std"])*100 / group["max_false"],
                (group["baseline_100"] + group["baseline_100_std"])*100 / group["max_false"],
                alpha=0.2
            )

        
        ax.axhline(100, color="black", linestyle="--", linewidth=2)

        ax.set_title(title_fn(label) + " (normalized)")
        ax.set_ylabel("normalized # persistent")
        ax.set_xlabel(x.replace("_", " ").title())

        if col == K - 1:
            ax.legend(title=legend_title, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    #saves the created plots to png files
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.tight_layout()
    #plt.show()

"""
Figure 1: x axis is truth fraction, y axis is # persistence(normalized + unnormalized)
each of the graphs shows a different network topology. Each line shows a trust range
"""
fixed_threshold = df["threshold"].iloc[0] #threshold was kept constant for this plot
df_f1 = df[(df["noise"] == fixed_noise) & (df["threshold"] == fixed_threshold)]

groups = [(net, df_f1[df_f1["network_type"] == net]) for net in df_f1["network_type"].unique()]
print(groups[0])

plot_maker(
    groups,
    x="truth_fraction",
    title_fn=lambda net: f"{net} Network",
    legend_title="Trust Range",
    legend = "trust_range",
    filename = "topologies"
)


"""
Figure 2: x axis is truth fraction, y axis is # persistence(normalized + unnormalized)
each of the graphs shows a different threshold. Each line shows a trust range
"""

df_ER = df[df["network_type"] == "ER"]
thresholds = sorted(df_ER["threshold"].unique())

groups = [(thr, df_ER[(df_ER["threshold"] == thr) & (df_ER["noise"] == fixed_noise)]) for thr in thresholds]

plot_maker(
    groups,
    x="truth_fraction",
    title_fn=lambda thr: f"ER Threshold={thr}",
    legend_title="Trust Range",
    legend = "trust_range",
    filename = "threshold"
)


"""
Figure 3: x axis is threshold, y axis is # persistence(normalized + unnormalized)
each of the graphs shows a different truth fraction. Each line shows a trust range
"""

truth_vals = sorted(df_ER["truth_fraction"].unique())

groups = [(t, df_ER[(df_ER["truth_fraction"] == t) & (df_ER["noise"] == fixed_noise)]) for t in truth_vals]

plot_maker(
    groups,
    x="threshold",
    title_fn=lambda t: f"ER Truth={t}",
    legend_title="Trust Range",
    legend = "trust_range",
    filename = "truth_fractions"
)


"""
Figure 4: x axis is truth fraction, y axis is # persistence(normalized + unnormalized)
each of the graphs shows a different trust ranges. Each line shows a different threshold
"""

trust_ranges = sorted(df_ER["trust_range"].unique())

groups = [(tr, df_ER[(df_ER["trust_range"] == tr) & (df_ER["noise"] == fixed_noise)]) for tr in trust_ranges]

plot_maker(
    groups,
    x="truth_fraction",
    title_fn=lambda tr: f"ER Trust={tr}",
    legend_title="Threshold",
    legend = "threshold",
    filename = "trust_ranges"
)

