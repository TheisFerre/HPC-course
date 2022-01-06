import pandas as pd
import matplotlib.pyplot as plt
import os

def add_data_plot(perm, opt, ax):
    file_name = perm + "_" + opt + "_matmult_c.gcc.dat"
    data = pd.read_csv(file_name, header=None, delim_whitespace=True)
    data.drop([2, 3], axis=1, inplace=True)
    data.columns = [
        "memory footprint in kB",
        "Mflop/s"
    ]

    #MFLOPS
    ax.plot(data["memory footprint in kB"], data["Mflop/s"], c=perm_color_dict[perm], marker="*", label=f"{perm}-{opt}")
    return ax

legend_order = [
    "mkn",
    "kmn",
    "nmk",
    "mnk",
    "knm",
    "nkm"
]

perm_color_dict = {
    "mkn": "blue",
    "kmn": "red",
    "nmk": "green",
    "mnk": "orange",
    "knm": "purple",
    "nkm": "brown"
}

opt_options = []
for f in os.listdir():
    if f.endswith(".dat"):
        opt = f.split("_")[1]
        if opt not in opt_options:
            opt_options.append(opt)

# Loop through different opt settings and creat a plot for each
for opt in opt_options:

    fig, ax = plt.subplots()

    for f in os.listdir():
        if f.endswith("dat") and opt in f:
            perm = f.split("_")[0]

            ax = add_data_plot(perm, opt, ax)

            #plt.yscale("log")
    """ax.vlines(x=12 * 32, ymin=0, ymax=1, color = 'b', linestyle="--")
    ax.text(x=12 * 32, y=2000, s="L1-cache")

    ax.vlines(x=12 * 256, ymin=0, ymax=1, color = 'b', linestyle="--")
    ax.text(x=12 * 256, y=2000, s="L2-cache")"""

    ax.set_xlabel("Memory footprint")
    ax.set_ylabel("Mflops/s")
    ax.set_xticklabels(ax.get_xticks(), rotation = 90)

    handles, labels = plt.gca().get_legend_handles_labels()


    ordering_handles = []
    ordering_labels = []
    for label_correct_order in [f"{handle}-{opt}" for handle in legend_order]:
        for i, label in enumerate(labels):
            if label == label_correct_order:
                ordering_handles.append(handles[i])
                ordering_labels.append(labels[i])


    ax.legend(ordering_handles, ordering_labels)
    plt.tight_layout()

    plt.savefig(f"{opt}-MFLOPS-comparison.png")

    



