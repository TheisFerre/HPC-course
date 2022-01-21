import pandas as pd
import matplotlib.pyplot as plt
import os

def load(filename):
    data = []
    thread_dataframes = {}

    # read stuff
    num = 1
    with open(filename, "r") as handle:
        for line in handle:
            if "#" in line:
                df = pd.DataFrame(data, columns=["KernelElapsed", "TransferElapsed"])
                df['Run'] = num
                thread_dataframes[num] = df
                data = []
                num += 1
            else:
                splitted = line.split()
                if len(splitted) < 2:
                    splitted = [splitted[0], None]
                data.append(splitted)

    # combine stuff
    combined = pd.concat(thread_dataframes.values(), ignore_index=True)
    combined = combined.apply(lambda col:pd.to_numeric(col, errors='coerce'))

    # mean value fusion
    result = combined.groupby('Run', axis=0, as_index=True, group_keys=True).mean().reset_index('Run')
    result['Ratio'] = result['TransferElapsed'] / result['KernelElapsed']
    return result

# get ready to plot
dfs = []
for file in os.listdir(os.getcwd()):
    if file.startswith("test_all") and file.endswith(".txt"):
        dfs.append(load(file))

sizes = ["64", "128", "256", "512", "1024", "2048", "4096"]

# KernelElapsed plot
fig, axs = plt.subplots(figsize=(12, 4))

results = {"GPU2":[], "GPU3":[], "GPU4":[], "GPU5":[], "GPULIB":[], "CPU":[]}
gpulib = []
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
gpulib = []
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
gpulib = []
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
