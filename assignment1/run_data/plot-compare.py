import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_dat(f_name):
    data = pd.read_csv(f_name, header=None, delim_whitespace=True)
    if f_name.startswith("blk"):
        data.drop([2, 3, 4], axis=1, inplace=True)
    else:
        data.drop([2, 3], axis=1, inplace=True)
    data.columns = [
        "memory footprint in kB",
        "Mflop/s"
    ]
    return data

# Loop through different opt settings and creat a plot for each

fig, ax = plt.subplots()
y_max = 0
y_max_new = 0

blk_dat = read_dat("blk_blk-compare_matmult_c.gcc.dat")
mkn_dat = read_dat("mkn_blk-compare_matmult_c.gcc.dat")


ax.plot(np.log(blk_dat["memory footprint in kB"]), blk_dat["Mflop/s"], marker="*", label=f"blk")
ax.plot(np.log(mkn_dat["memory footprint in kB"]), mkn_dat["Mflop/s"], marker="*", label=f"mkn")

ax.axvline(x=np.log(32), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log(32), y=max(mkn_dat["Mflop/s"]) * (1/3), s="L1-cache", c="black")

ax.axvline(x=np.log(256), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log(256), y=max(mkn_dat["Mflop/s"]) * (2/3), s="L2-cache")  

ax.axvline(x=np.log(30000), ymin=0, ymax=1, color = 'black', linestyle="--")
ax.text(x=np.log(30000), y=max(mkn_dat["Mflop/s"]) * (3/3), s="L3-cache")

ax.set_xlabel("Memory footprint in kB")
ax.set_ylabel("Mflops/s")
ax.set_ylim(0, None)
x_dat = plt.gca().get_lines()[0].get_xdata()
ax.set_xticklabels([int(dat) for dat in np.exp(x_dat)], rotation = 90, fontsize="8")
#ax.set_xticklabels(blk_dat["memory footprint in kB"], rotation = 90)
ax.set_xticks(np.log(blk_dat["memory footprint in kB"]))
ax.set_title("Mflops/s vs. Memory footprint in kB\nBlocksize=128")

ax.legend()
plt.tight_layout()

plt.savefig(f"COMPARE.png")

