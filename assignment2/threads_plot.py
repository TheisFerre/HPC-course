import pandas as pd
import matplotlib.pyplot as plt

jacobi_f_name = "jacobi-thread-runtime.dat"
data_jacobi = pd.read_csv(jacobi_f_name, sep=" ", header=None, usecols=[0,1])

jacobi_opt_f_name = "jacobi-opt-thread-runtime.dat"
data_opt_jacobi = pd.read_csv(jacobi_opt_f_name, sep=" ", header=None, usecols=[0,1])

gauss_f_name = "gauss-thread-runtime.dat"
data_gauss = pd.read_csv(gauss_f_name, sep=" ", header=None, usecols=[0,1])


cols  = [
    "threads",
    "runtime"
]

data_jacobi.columns = cols
data_opt_jacobi.columns = cols
data_gauss.columns = cols

def calc_scaleup(df):
    dat = []
    for idx, row in df.iterrows():
        if idx == 0:
            base_val = row["runtime"]
        dat.append(base_val/row["runtime"])
    return dat

data_jacobi["scaleup"] = calc_scaleup(data_jacobi)
data_opt_jacobi["scaleup"] = calc_scaleup(data_opt_jacobi)
data_gauss["scaleup"] = calc_scaleup(data_gauss)


plt.plot(data_jacobi["threads"], data_jacobi["scaleup"], label="Jacobi (naive impl.)")
plt.plot(data_opt_jacobi["threads"], data_opt_jacobi["scaleup"], label="Jacobi (optimized impl.)")
plt.plot(data_gauss["threads"], data_gauss["scaleup"], label="Gauss-Seidel")
plt.plot(range(1, len(data_gauss) + 1), range(1, len(data_gauss) + 1), linestyle="dashed", label="Perfect scaling")
plt.ylabel("scaleup")
plt.xlabel("Threads")
plt.xticks(list(range(1, len(data_gauss) + 1)))
plt.title("scaling vs. threads")
plt.legend()
plt.tight_layout()
plt.savefig("threads-plot.png")

