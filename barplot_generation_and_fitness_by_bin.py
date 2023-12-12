import json
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def plot(data_file, bins, suffix="plot"):
    with open(data_file, 'r') as file:
        json_data = file.readlines()

    data = [json.loads(line) for line in json_data]

    print(json.dumps(data[0], indent=4))

    # iteration steps, eg. 100 ... 1900
    iterations = [100*(i+1) for i in range(int((len(data))/100))]

    plt_bins = [0]
    for b in bins: plt_bins.append(b)
    plt_bins.append(1)

    data_binned_fitness = {iteration: [[] for bin in range(len(bins)+1)] for iteration in iterations}
    data_binned_generations = {iteration: [0 for bin in range(len(bins)+1)] for iteration in iterations}
    data_binned_fitness_mean = {iteration: [0 for bin in range(len(bins)+1)] for iteration in iterations}
    data_binned_fitness_max = {iteration: [0 for bin in range(len(bins)+1)] for iteration in iterations}

    for item in data:
        # skip iterations without solution generated
        if item["phenotype"] is None:
            continue

        # bin for this data point
        bin = 0
        for b in bins:
            # only 1d axis plotting is supported for this script, change the index corresponding to diversity measure for plot
            if item["phenotype"][0] > b:
                bin += 1
        
        # loop over iterations to fill those histograms
        for iter in iterations:
            data_binned_generations[iter][bin] += 1
            data_binned_fitness[iter][bin].append(item["fitness"])

    # max fitness
    for iteration in data_binned_fitness:
        for idx, bin_data in enumerate(data_binned_fitness[iteration]):
            if len(bin_data) > 0:
                data_binned_fitness_max[iteration][idx] = np.amax(bin_data)

    fig = plt.figure(figsize=(8,6))
    for i in reversed(iterations):
        if i % 2 == 0:
            plt.bar(range(20), data_binned_generations[i], color="b", alpha=0.05, edgecolor='black')
    plt.xlabel("Bin Index")
    plt.xticks(range(0,20,5))
    plt.xlim(-0.5, 19.5)
    plt.ylabel("Number of Generations")
    plt.title("Number of entries generated per bin in 100 iterations steps")
    plt.show()
    fig.savefig(f'generation_per_bin_{suffix}.png')

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    for i in reversed(iterations):
        if i % 2 == 0:
            plt.bar(range(20), data_binned_fitness_max[i], color="r", alpha=0.05, edgecolor='black')
    plt.xlabel("Bin Index")
    plt.ylabel("Fitness")
    plt.xlim(-0.5, 19.5)
    plt.xticks(range(0,20,5))
    plt.ylim(0.9, 1)
    plt.title("Max fitness per bin in 100 iterations steps")
    plt.show()

    fig.savefig(f'fitness_per_bin_{suffix}.png')

if __name__ == "__main__":
    # 20 bins cover range [0,1] - change to described 10-bin setting for experiments using 2D archive
    plot(
        data_file = "data/histories_opinions_stories/qdaif/opinions/lmx_near_seeded_init/1/history.jsonl",
        bins = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995],
        suffix = "1D",
    )
