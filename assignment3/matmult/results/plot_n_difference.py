# parse threads compare

import pandas as pd
import matplotlib.pyplot as plt

data = []
thread_dataframes = {}

# read stuff
num = 1
with open("test_compare_2048.txt", "r") as handle:
    for line in handle:
        if "#" in line:
            df = pd.DataFrame(data, columns=["Kernel", "Transfer"])
            df['Num'] = num
            thread_dataframes[num] = df
            data = []
            num += 1
        else:
            data.append(line.split())

# combine stuff
combined = pd.concat(thread_dataframes.values(), ignore_index=True)
combined = combined.apply(lambda col:pd.to_numeric(col, errors='coerce'))

# reset group into normal dataframe
result = combined.groupby('Num', axis=0, as_index=True, group_keys=True).mean().reset_index('Num')

# plot that badboy
ax = result.plot.bar(x='Num', y='Kernel', rot=0)

plt.ylabel("Avg. Kernel wallclock time")
plt.xlabel("N")
plt.title("Thread compute N elements")
plt.tight_layout()
plt.legend()
plt.savefig("n_difference.png")