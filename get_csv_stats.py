import json
import numpy as np
from pathlib import Path
from typing import Any
import csv
from sklearn.utils import resample

def mean_fn(data):
    return np.mean(data)

def bootstrap_ci(data, mean_method, num_samples=100000, alpha=0.05):
    n = len(data)
    means = [mean_method(resample(data)) for _ in range(num_samples)]
    print("bootstrap resample done")
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper

def get_last_point_stats(data):
    data = np.array(data)

    mean = np.mean(data, axis=0)[-1]
    std_error = np.std(data, axis=0)[-1] / np.sqrt(data.shape[0])
    return mean, std_error

def load_elites_stats(
    base_dirs: list[str],
) -> dict[str, Any]:
    qd_score_runs = {}
    max_fitness_runs = {}
    coverage_runs = {}
    prev_expt_dir = ""

    # contains every experiment of runs in base_dir
    for base_dir in base_dirs:
        # sort so that experiments are processed at a time
        for sub_dir in sorted(base_dir.rglob("elites.jsonl")):
            expt_dir = sub_dir.parent.parent # this dir would contain multiple seed runs
            # sub_dir goes through all rerun dir elites for an expt dir, so we continue onto next expt dir 
            if expt_dir == prev_expt_dir:
                continue
            
            # initialize to store stats across reruns in single experiment
            qd_score_seeds = []
            max_fitness_seeds = []
            coverage_seeds = []

            # process all random rng seed reruns in experiment
            for seed_dir in expt_dir.rglob("elites.jsonl"):
                last_iteration_number = -1
                bins_last_iteration = []
                
                # for end of run stats
                with open(seed_dir, "rb") as f:
                    for line in f:
                        data = json.loads(line)
                        if data['iteration'] > last_iteration_number:
                            last_iteration_number = data['iteration']


                # gather all bins from last iteration
                with open(seed_dir, "rb") as f:
                    for line in f:
                        data = json.loads(line)
                        if data['iteration'] == last_iteration_number:
                            bins_last_iteration.append(data)

                # coverage
                non_empty_bins = [bin_data for bin_data in bins_last_iteration if bin_data['elite'].strip() != ""]
                coverage_seed = (len(non_empty_bins) / len(bins_last_iteration))

                # score of max fitness solution across all iterations
                highest_fitness_score_seed = max([bin_data['fitness'] for bin_data in bins_last_iteration], default=0)

                # sum of the fitness scores across bins in final state
                qd_score_seed = sum([bin_data['fitness'] for bin_data in bins_last_iteration if bin_data['elite'].strip() != ""])

                qd_score_seeds.append([qd_score_seed])
                max_fitness_seeds.append([highest_fitness_score_seed])
                coverage_seeds.append([coverage_seed])

            # store stats for each experiment, to be accessed by key for the experiment (absolute) path
            qd_score_runs[str(expt_dir)] = qd_score_seeds
            max_fitness_runs[str(expt_dir)] = max_fitness_seeds
            coverage_runs[str(expt_dir)] = coverage_seeds
            prev_expt_dir = expt_dir
    return qd_score_runs, max_fitness_runs, coverage_runs

if __name__ == "__main__":
    # define directory (set) containing all/multiple experiments, or just individual experiments (with 5 re-runs) 
    runs_whitelist = [
        "data/histories_opinions_stories/qdaif/opinions",
        "data/histories_opinions_stories/qdaif/stories_ending/lmx_near_seeded_init",
        "data/histories_opinions_stories/qdaif/stories_ending/lmx_near_zero_init",
    ]

    csv_output_file = "qd_score.csv"

    qd_score_runs, max_fitness_runs, coverage_runs = load_elites_stats([Path(runs_whitelist[i]) for i in range(len(runs_whitelist))])
    data_for_csv = qd_score_runs # choose stat to compute for output CSV (qd score, max fitness, coverage)

    with open(csv_output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Run Directory", "Last Point Mean", "Last Point Standard Error", "Mean Bootstrap CI Lower", "Mean Bootstrap CI Higher"])
        
        # compute stats for each run
        for run in sorted(data_for_csv.keys()):
            expt_data = data_for_csv[run]
            last_point_mean, last_point_std_error = get_last_point_stats(expt_data)

            expt_data = data_for_csv[run]

            last_point_data = [seed[-1] for seed in expt_data]

            # 95% CI
            mean_ci_lower, mean_ci_upper = bootstrap_ci(last_point_data, mean_fn)
            
            csvwriter.writerow([run, last_point_mean, last_point_std_error, mean_ci_lower, mean_ci_upper])

