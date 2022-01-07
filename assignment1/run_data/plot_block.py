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

ax.plot(np.log2(data["blocksize"]), data["Mflop/s"], marker="*", label=f"Mflops/s")

ax.axvline(x=np.log2(36), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log2(36), y=max(data["Mflop/s"]) * (1/3), s="L1-cache", c="black")

ax.axvline(x=np.log2(103), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log2(103), y=max(data["Mflop/s"]) * (2/3), s="L2-cache")

ax.axvline(x=np.log2(1131), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log2(1131), y=max(data["Mflop/s"]) * (2/3), s="L3-cache")  

ax.set_xlabel("blocksize")
ax.set_ylabel("Mflops/s")
ax.set_ylim(0, None)
ax.set_xticklabels(data["blocksize"], rotation = 90)
ax.set_xticks(np.log2(data["blocksize"]))
ax.set_title("Mflops/s vs. blocksize For M=N=K=2000")

ax.legend()
plt.tight_layout()

plt.savefig(f"block-MFLOPS-comparison.png")




