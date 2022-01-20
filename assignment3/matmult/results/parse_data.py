import pandas as pd

in_file = open("test_compare_2048.txt", "r")
data = []

for line in in_file:
    if "#" in line:
        df = pd.DataFrame(data, columns=["Kernel", "Transfer"])
        print(df)
    else:
        data.append(line.split())
