import pandas as pd
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
                df['MegaFlops'] = line.split()[1]
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