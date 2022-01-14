import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


jacobi_f = "scaling-jacobi-runtime.dat"
gauss_f = "scaling-gauss-runtime.dat"

jacobi_data = pd.read_csv(jacobi_f, header=None, sep=" ")
gauss_data = pd.read_csv(gauss_f, header=None, sep=" ")

cols = ["threads", "N", "runtime"]

jacobi_data.columns = cols
gauss_data.columns = cols

cmap = get_cmap("tab20")
colors = cmap.colors


N_vals = [400, 100, 50, 10]  #[500, 400, 300, 200, 100, 50]

counter = 0

for n in N_vals:

    jacobi_subset = jacobi_data[jacobi_data["N"] == n]
    gauss_subset = gauss_data[jacobi_data["N"] == n]

    plt.plot(gauss_subset["threads"], gauss_subset["runtime"], c=colors[counter], marker="*", linestyle="dashed", label=f"gauss-seidel (N={n})")
    plt.plot(jacobi_subset["threads"], jacobi_subset["runtime"], c=colors[counter+1], marker="*", label=f"jacobi (N={n})")
    

    counter += 2

plt.ylabel("Runtime (s)")
plt.yscale("log")
plt.xlabel("Threads")
plt.xticks(list(range(1, 8+1)))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Scaling comparison of methods\n Runtime(s) vs. threads")
plt.tight_layout()
plt.savefig("scaling-plot.png")











