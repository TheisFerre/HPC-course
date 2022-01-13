import pandas as pd
import matplotlib.pyplot as plt

jacobi_f_name = "poisson_runtime.dat"
data_jacobi = pd.read_csv(jacobi_f_name, sep=" ")

data_jacobi.columns = [
    "k",
    "time",
    "N"
]

data_jacobi["Mlups/s"] = data_jacobi["N"]**3 * data_jacobi["k"] / 1000000

plt.plot(data_jacobi["N"], data_jacobi["Mlups/s"], label="Jacobi method")
plt.ylabel("Mlups/s")
plt.xlabel("N")
plt.title("Mlups/s comparison")
plt.legend()
plt.tight_layout()
plt.savefig("runtime-plot.png")
