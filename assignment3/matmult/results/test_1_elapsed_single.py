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
                print(line)
                splitted = line.split()
                if len(splitted) < 2:
                    splitted = [splitted[0], None]
                data.append(splitted)

    # combine stuff
    combined = pd.concat(thread_dataframes.values(), ignore_index=True)
    combined = combined.apply(lambda col:pd.to_numeric(col, errors='coerce'))

    # mean value fusion
    result = combined.groupby('Run', axis=0, as_index=True, group_keys=True).mean().reset_index('Run')
    return result

# get ready to plot
dfs = []
for file in os.listdir(os.getcwd()):
    if file.startswith("test_1_single") and file.endswith(".txt"):
        dfs.append(load(file))

# KernelElapsed plot

# TransferElapsed plot


