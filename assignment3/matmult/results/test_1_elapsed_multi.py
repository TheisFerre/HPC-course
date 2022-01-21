import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import load

# get ready to plot
dfs = []
for file in os.listdir(os.getcwd()):
    if file.startswith("test_1_mult") and file.endswith(".txt"):
        dfs.append(load(file))

sizes = ["64", "128", "256", "512", "1024", "2048"]

# KernelElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))
gpu = []
cpu = []
for df in dfs:
    gpu.append(df[df['Run'] == 1].KernelElapsed.values[0])
    cpu.append(df[df['Run'] == 2].KernelElapsed.values[0])

plt.plot(sizes,gpu,marker="o",label='GPU2')
plt.plot(sizes,cpu,marker="o",label="CPU")
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel("Avg. Kernel Elapsed time sec.")
plt.xlabel("Matrix Size")
plt.title("Kernel Runtime")
plt.tight_layout()
plt.savefig("Test_1_KernelElapsed_Multi.png")

# TransferElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))
gpu = []
for df in dfs:
    gpu.append(df[df['Run'] == 1].TransferElapsed.values[0])

plt.plot(sizes,gpu,marker="o",label='GPU2')
# plt.plot(sizes,cpu,marker="o",label="CPU")
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel("Avg. Transfer Elapsed time sec.")
plt.xlabel("Matrix Size")
plt.title("Transfer Runtime")
plt.tight_layout()
plt.savefig("Test_1_TransferElapsed_Multi.png")

# Transfer / Kernel Ratio
fig, axs = plt.subplots(figsize=(12, 4))
gpu = []
for df in dfs:
    gpu.append(df[df['Run'] == 1].Ratio.values[0])

plt.plot(sizes,gpu,marker="o",label='GPU2')
plt.legend()
plt.grid()
# plt.yscale('log')
plt.ylabel("Ratio")
plt.xlabel("Matrix Size")
plt.title("Transfer / Kernel Ratio")
plt.tight_layout()
plt.savefig("Test_1_Ratio_Multi.png")

# MegaFlops
fig, axs = plt.subplots(figsize=(12, 4))

results = {"GPU2":[], "CPU":[]}
c = 0
for df in dfs:
    l = []
    i = 0
    for key, val in results.items():
        results[key].append(df.at[i,'MegaFlops'])
        i += 1

for key, val in results.items():
    plt.plot(sizes,results[key],marker="o",label=key)

plt.legend()
plt.grid()
# plt.yscale('log')
plt.ylabel("MFlops/s")
plt.xlabel("Matrix Size")
plt.title("MFlops Performance")
plt.tight_layout()
plt.savefig("Test_1_MegaFlops_multi.png")