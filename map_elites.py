from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

"""
Assesses the performance stats of any run (QDAIF, baselines) based on the logged iterations of texts and their measures from jsonl
"""

class MAPElitesEvaluation:
    def __init__(self, history_length: int, n_bins: int, start:float, stop:float, custom_bins=None, elites_intervals=100):
        self.n_bins = n_bins
        self.history_length = history_length
        self.archive = np.full((history_length, n_bins), -np.inf, dtype=np.float32)
        if custom_bins is not None:
            self.bins = custom_bins
        else:
            self.bins = np.linspace(start, stop, n_bins+1)[1:]
        self.elites = n_bins*[""]
        self.elites_intervals = elites_intervals

    def fit(self, phenotype_key: str, data: pd.DataFrame, elites_path: Optional[Path] = None):
        assert self.history_length <= len(data["genotype"]), "retry script with history_length within number of total available data points"
        
        for i, (genotype, phenotype, fitness) in tqdm(enumerate(zip(data["genotype"], data[phenotype_key], data["fitness"]))):
            if phenotype is None or phenotype[0] is None:
                continue
            
            bin_idx = np.digitize(phenotype[0], self.bins)

            if self.archive[i, bin_idx] < fitness:
                self.archive[i:, bin_idx] = fitness
                self.elites[bin_idx] = genotype

            # print elite stats
            if (i%self.elites_intervals == 0 or i == (len(data["genotype"])-1)) and i > 0 and elites_path is not None:
                with open(elites_path, "a+", encoding="utf-8") as f:
                    for k, (fit, elite) in enumerate(zip(self.archive[i], self.elites)):
                        json_record = json.dumps({"iteration":i, "bin_idx":k, "elite": elite, "fitness": float(fit)}, ensure_ascii=False)
                        f.write(json_record + "\n")

            # end if past desired history length (iterations)
            if i >= self.history_length - 1:
                break

    @property
    def max_fitnesses(self):
        return self.archive.max(1)

    @property
    def min_fitnesses(self):
        archive_copy = np.copy(self.archive)
        archive_copy[archive_copy == -np.inf] = np.inf
        return archive_copy.min(1)

    @property
    def mean_fitnesses(self):
        return self.qd_scores / self.n_bins

    @property
    def qd_scores(self):
        archive_copy = np.copy(self.archive)
        archive_copy[archive_copy == -np.inf] = 0
        return archive_copy.sum(1)
    
    @property
    def coverage(self):
        # Count the number of non-empty bins (-np.inf indicates an empty bin)
        non_empty_bins = np.not_equal(self.archive, -np.inf).astype(int)
        
        # Calculate coverage for each iteration
        coverage_per_iteration = non_empty_bins.sum(axis=1) / self.n_bins

        return coverage_per_iteration


class MAPElites2DEvaluation:
    def __init__(self, history_length: int, x_bins: Union[int, List[float]], y_bins: Union[int, List[float]], start:Tuple[float], stop:Tuple[float], elites_intervals=100):

        self.n_x_bins = x_bins if isinstance(x_bins, int) else len(x_bins) + 1
        self.n_y_bins = y_bins if isinstance(y_bins, int) else len(y_bins) + 1

        self.history_length = history_length

        self.archive = np.full((history_length, self.n_x_bins, self.n_y_bins), -np.inf, dtype=np.float32)
        self.bins_x = np.linspace(start[0], stop[0], self.n_x_bins[0]+1)[1:-1] if isinstance(x_bins, int) else np.array(x_bins)
        self.bins_y = np.linspace(start[1], stop[1], self.n_y_bins[1]+1)[1:-1] if isinstance(y_bins, int) else np.array(y_bins)
        self.elites = [["" for i in range(self.n_x_bins)] for j in range(self.n_y_bins)]
        self.elites_intervals = elites_intervals

    def fit(self, phenotype_key: str, data: pd.DataFrame, elites_path: Optional[Path] = None):
        assert self.history_length <= len(data["genotype"]), "retry script with history_length within number of total available data points"
        
        for i, (genotype, phenotype, fitness) in tqdm(enumerate(zip(data["genotype"], data[phenotype_key], data["fitness"]))):
            if phenotype is None:
                continue
            phenotype_x, phenotype_y = phenotype
            if phenotype_x is None or phenotype_y is None or fitness is None:
                continue

            bin_idx_x = np.digitize(phenotype_x, self.bins_x)
            bin_idx_y = np.digitize(phenotype_y, self.bins_y)

            if self.archive[i, bin_idx_x, bin_idx_y] < fitness:
                self.archive[i:, bin_idx_x, bin_idx_y] = fitness
                self.elites[bin_idx_x][bin_idx_y] = genotype

            # print elite stats
            if (i%self.elites_intervals == 0 or i == (self.history_length-1)) and i > 0 and elites_path is not None:
                with open(elites_path, "a+", encoding="utf-8") as f:
                    for k, (fit, elite) in enumerate(zip(self.archive[i], self.elites)):
                        for j, (fit_2, elite_2) in enumerate(zip(fit, elite)):
                            json_record = json.dumps({"iteration":i, "bin_idx_0":k, "bin_idx_1":j, "elite": elite_2, "fitness": float(fit_2)}, ensure_ascii=False)
                            f.write(json_record + "\n")
            
            # end if past desired history length (iterations)
            if i >= self.history_length - 1:
                break

    def get_bin_index(self, phenotype: List[float]):
        return np.array([np.digitize(phenotype[0], self.bins_x), np.digitize(phenotype[1], self.bins_x)])

    @property
    def max_fitnesses(self):
        return self.archive.max((1, 2))

    @property
    def min_fitnesses(self):
        archive_copy = np.copy(self.archive)
        archive_copy[archive_copy == -np.inf] = np.inf
        return archive_copy.min((1, 2))

    @property
    def mean_fitnesses(self):
        return self.qd_scores / self.n_x_bins / self.n_y_bins

    @property
    def qd_scores(self):
        archive_copy = np.copy(self.archive)
        archive_copy[archive_copy == -np.inf] = 0
        return archive_copy.sum((1, 2))
    
    @property
    def coverage(self):
        # Count the number of non-empty bins (-np.inf indicates an empty bin)
        non_empty_bins = np.not_equal(self.archive, -np.inf).astype(int)
        
        # Calculate coverage for each iteration
        coverage_per_iteration = non_empty_bins.sum(axis=(1,2)) / (self.n_x_bins * self.n_y_bins)

        return coverage_per_iteration