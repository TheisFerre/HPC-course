import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


MAT_SIZE = 2048
file_name = "blk-fast-unroll_matmult_c.gcc.dat"


data = pd.read_csv(file_name, header=None, delim_whitespace=True)
data.drop([0, 2, 3], axis=1, inplace=True)
data.columns = [
    "Mflop/s",
    "blocksize"
]

fig, ax = plt.subplots()

ax.plot(np.log(data["blocksize"]), data["Mflop/s"], marker="*", label=f"Mflops/s")

ax.axvline(x=np.log(32), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log(32), y=max(data["Mflop/s"]) * (1/3), s="L1-cache", c="black")

ax.axvline(x=np.log(256), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log(256), y=max(data["Mflop/s"]) * (2/3), s="L2-cache")  

ax.set_xlabel("log(blocksize)")
ax.set_ylabel("Mflops/s")
ax.set_ylim(0, None)
ax.set_xticklabels(ax.get_xticks(), rotation = 90)
ax.set_title("Mflops/s vs. log(blocksize)")

ax.legend()
plt.tight_layout()

plt.savefig(f"block-MFLOPS-comparison.png")




