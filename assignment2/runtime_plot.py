import pandas as pd
import matplotlib.pyplot as plt

jacobi_f_name = "poisson_jacobi_runtime.dat"
data_jacobi = pd.read_csv(jacobi_f_name, sep=" ")

gauss_f_name = "poisson_gauss_runtime.dat"
data_gauss = pd.read_csv(gauss_f_name, sep=" ")

cols  = [
    "k",
    "time",
    "N"
]

data_jacobi.columns = cols
data_gauss.columns = cols


data_jacobi["Mlups/s"] = data_jacobi["N"]**3 * data_jacobi["k"] / 1000000
data_gauss["Mlups/s"] = data_gauss["N"]**3 * data_gauss["k"] / 1000000

plt.plot(data_jacobi["N"], data_jacobi["Mlups/s"], label="Jacobi method")
plt.plot(data_gauss["N"], data_gauss["Mlups/s"], label="Jacobi method")
plt.ylabel("Mlups/s")
plt.xlabel("N")
plt.title("Mlups/s comparison")
plt.legend()
plt.tight_layout()
plt.savefig("runtime-plot.png")
