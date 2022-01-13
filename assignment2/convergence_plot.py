import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("poisson_jacobi_convergence.dat")
data.columns = {'norm'}

ax = data.plot.line()

ax.set_xlabel("Iteration")
ax.set_ylabel("Frobenius Norm")
ax.set_ylim(0, None)
ax.set_title("Convergence")

ax.legend()
plt.tight_layout()

plt.savefig("convergence.png")