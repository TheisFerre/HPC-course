import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import load

# get ready to plot
dfs = []
for file in os.listdir(os.getcwd()):
    if file.startswith("test_all") and file.endswith(".txt"):
        dfs.append(load(file))

sizes = ["64", "128", "256", "512", "1024", "2048", "4096"]

# KernelElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))

results = {"GPU2":[], "GPU3":[], "GPU4":[], "GPU5":[], "GPULIB":[], "CPU":[]}
c = 0
for df in dfs:
    l = []
    i = 0
    for key, val in results.items():
        results[key].append(df.at[i,'KernelElapsed'])
        i += 1

for key, val in results.items():
    plt.plot(sizes,results[key],marker="o",label=key)

plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel("Avg. Kernel Elapsed time sec.")
plt.xlabel("Matrix Size")
plt.title("Kernel Runtime")
plt.tight_layout()
plt.savefig("Test_All_KernelElapsed.png")

# TransferElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))

results = {"GPU2":[], "GPU3":[], "GPU4":[], "GPU5":[], "GPULIB":[]}
c = 0
for df in dfs:
    l = []
    i = 0
    for key, val in results.items():
        results[key].append(df.at[i,'TransferElapsed'])
        i += 1

for key, val in results.items():
    plt.plot(sizes,results[key],marker="o",label=key)

# plt.plot(sizes,cpu,marker="o",label="CPU")
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel("Avg. Transfer Elapsed time sec.")
plt.xlabel("Matrix Size")
plt.title("Transfer Runtime")
plt.tight_layout()
plt.savefig("Test_All_TransferElapsed.png")

# Transfer / Kernel Ratio
fig, axs = plt.subplots(figsize=(12, 4))

results = {"GPU2":[], "GPU3":[], "GPU4":[], "GPU5":[], "GPULIB":[]}
c = 0
for df in dfs:
    l = []
    i = 0
    for key, val in results.items():
        results[key].append(df.at[i,'Ratio'])
        i += 1

for key, val in results.items():
    plt.plot(sizes,results[key],marker="o",label=key)

plt.legend()
plt.grid()
# plt.yscale('log')
plt.ylabel("Ratio")
plt.xlabel("Matrix Size")
plt.title("Transfer / Kernel Ratio")
plt.tight_layout()
plt.savefig("Test_All_Ratio.png")

# MegaFlops
fig, axs = plt.subplots(figsize=(12, 4))

results = {"GPU2":[], "GPU3":[], "GPU4":[], "GPU5":[], "GPULIB":[], "CPU":[]}
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
plt.savefig("Test_All_MegaFlops.png")