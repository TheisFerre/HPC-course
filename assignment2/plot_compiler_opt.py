import pandas as pd
import matplotlib.pyplot as plt

jacobi_f_name = "jacobi-unoptimized-runtime.dat"
data_jacobi = pd.read_csv(jacobi_f_name, sep=" ", header=None, usecols=[0,1])

jacobi_opt_f_name = "jacobi-optimized-runtime.dat"
data_opt_jacobi = pd.read_csv(jacobi_opt_f_name, sep=" ", header=None, usecols=[0,1])


cols  = [
    "threads",
    "runtime"
]

data_jacobi.columns = cols
data_opt_jacobi.columns = cols

def calc_scaleup(df):
    dat = []
    for idx, row in df.iterrows():
        if idx == 0:
            base_val = row["runtime"]
        dat.append(base_val/row["runtime"])
    return dat

data_jacobi["scaleup"] = calc_scaleup(data_jacobi)
data_opt_jacobi["scaleup"] = calc_scaleup(data_opt_jacobi)


plt.plot(data_jacobi["threads"], data_jacobi["scaleup"], label="Non-optimized")
plt.plot(data_opt_jacobi["threads"], data_opt_jacobi["scaleup"], label="Optimized")
plt.plot(range(1, len(data_opt_jacobi) + 1), range(1, len(data_opt_jacobi) + 1), linestyle="dashed", label="Perfect scaling")
plt.ylabel("scaleup")
plt.xlabel("Threads")
plt.xticks(list(range(1, len(data_opt_jacobi) + 1)))
plt.title("Compiler optimization influence")
plt.legend()
plt.tight_layout()
plt.savefig("comp-opt-plot.png")

