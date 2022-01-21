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
    if file.startswith("test_2") and file.endswith(".txt"):
        dfs.append(load(file))

sizes = ["64", "128", "256", "512", "1024", "2048", "4096"]

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
