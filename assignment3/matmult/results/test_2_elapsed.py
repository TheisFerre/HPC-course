import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from utils import load

# get ready to plot
# dfs = []
# for file in os.listdir(os.getcwd()):
#     if file.startswith("test_2") and file.endswith(".txt"):
#         dfs.append(load(file))

sizes = ["64", "128", "256", "512", "1024", "2048", "4096"]

candidates = []
for file in os.listdir(os.getcwd()):
    if file.startswith("test_2_") and file.endswith(".txt"):
        candidates.append(file)
dfs = []
for file in sorted(candidates, key=lambda x:int(re.findall("(\d+)",x)[1])):
    print(file)
    dfs.append(load(file))

# KernelElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))
gpu3 = []
gpu4 = []
cpu = []
for df in dfs:
    gpu3.append(df[df['Run'] == 1].KernelElapsed.values[0])
    gpu4.append(df[df['Run'] == 2].KernelElapsed.values[0])
    cpu.append(df[df['Run'] == 3].KernelElapsed.values[0])

plt.plot(sizes,gpu3,marker="o",label='GPU3')
plt.plot(sizes,gpu4,marker="o",label='GPU4')
plt.plot(sizes,cpu,marker="o",label="CPU")
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel("Avg. Kernel Elapsed time sec.")
plt.xlabel("Matrix Size")
plt.title("Kernel Runtime")
plt.tight_layout()
plt.savefig("Test_2_KernelElapsed.png")

# TransferElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))
gpu3 = []
gpu4 = []
for df in dfs:
    gpu3.append(df[df['Run'] == 1].TransferElapsed.values[0])
    gpu4.append(df[df['Run'] == 2].TransferElapsed.values[0])

plt.plot(sizes,gpu3,marker="o",label='GPU3')
plt.plot(sizes,gpu4,marker="o",label='GPU4')
plt.legend()
plt.grid()
plt.yscale('log')
plt.ylabel("Avg. Transfer Elapsed time sec.")
plt.xlabel("Matrix Size")
plt.title("Transfer Runtime")
plt.tight_layout()
plt.savefig("Test_2_TransferElapsed.png")

# Transfer / Kernel Ratio
fig, axs = plt.subplots(figsize=(12, 4))
gpu3 = []
gpu4 = []
for df in dfs:
    gpu3.append(df[df['Run'] == 1].Ratio.values[0])
    gpu4.append(df[df['Run'] == 2].Ratio.values[0])

print(gpu3)
print(gpu4)

plt.plot(sizes,gpu3,marker="o",label='GPU3')
plt.plot(sizes,gpu4,marker="o",label='GPU4')
plt.legend()
plt.grid()
plt.ylabel("Ratio")
plt.xlabel("Matrix Size")
plt.title("Transfer / Kernel Ratio")
plt.tight_layout()
plt.savefig("Test_2_Ratio.png")

# # MegaFlops
# fig, axs = plt.subplots(figsize=(12, 4))

# results = {"GPU3":[], "GPU4":[], "CPU":[]}
# c = 0
# for df in dfs:
#     l = []
#     i = 0
#     for key, val in results.items():
#         results[key].append(df.at[i,'MegaFlops'])
#         i += 1

# for key, val in results.items():
#     plt.plot(sizes,results[key],marker="o",label=key)

# plt.legend()
# plt.grid()
# # plt.yscale('log')
# plt.ylabel("MFlops/s")
# plt.xlabel("Matrix Size")
# plt.title("MFlops Performance")
# plt.tight_layout()
# plt.savefig("Test_2_MegaFlops.png")