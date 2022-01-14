import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

jacobi_f_name = "poisson_jacobi_convergence.dat"
data_jacobi = pd.read_csv(jacobi_f_name, sep=" ")

gauss_f_name = "poisson_gauss_convergence.dat"
data_gauss = pd.read_csv(gauss_f_name, sep=" ")

data_jacobi.columns = {'norm'}
data_gauss.columns = {'norm'}

fig, ax = plt.subplots()

ax.plot(range(1, len(data_jacobi) + 1), data_jacobi["norm"], label="Jacobi method")
ax.plot(range(1, len(data_gauss) + 1), data_gauss["norm"], label="Gauss-Seidel method")
ax.axhline(y=1, c="r", linestyle="dashed", label="Tolerance=1")

ax.set_xlabel("Iteration")
ax.set_ylabel("Frobenius Norm")
ax.set_yscale("log")
ax.set_ylim(0, None)
ax.set_title("Convergence")

ax.legend()
plt.tight_layout()

plt.savefig("convergence.png")
