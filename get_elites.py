import numpy as np
import pandas as pd
from pathlib import Path
from map_elites import MAPElitesEvaluation, MAPElites2DEvaluation

RESULT_PATHS = [
    "data/histories_opinions_stories",
]
N_BINS = 20 # number of bin intervals
ELITES_INTERVALS = 100 # elites at a given archive state saved every N intervals

if __name__ == "__main__":
    for base_path in RESULT_PATHS:
        for path in Path(base_path).rglob("history.jsonl"):
            print(path)
            X = pd.read_json(path, lines=True)
            history_length = len(X) # set it to a shorter length of iterations if desired

            elites_path = Path(path).parent / "elites.jsonl" # output path for logged elite solutions for each bin

            if "qdef/" in str(elites_path): # for embedding feedback experiment logs
                custom_bins = np.array([0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60])
            else: # default
                custom_bins = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995])

            if "stories_genre_ending" in str(elites_path): # 2D grid domain
                X_BINS = [0.005, 0.02, 0.05, 0.20, 0.50, 0.80, 0.95, 0.98, 0.995]
                Y_BINS = [0.005, 0.02, 0.05, 0.20, 0.50, 0.80, 0.95, 0.98, 0.995]

                map_elites_evaluation = MAPElites2DEvaluation(history_length=history_length, x_bins=X_BINS, y_bins=Y_BINS, start=(0,0), stop=(1,1), elites_intervals=ELITES_INTERVALS)
                map_elites_evaluation.fit(phenotype_key="phenotype", data=X, elites_path=elites_path)
            else:
                # pass
                map_elites_evaluation = MAPElitesEvaluation(history_length=history_length, n_bins=N_BINS, start=0, stop=1, custom_bins=custom_bins, elites_intervals=ELITES_INTERVALS)
                map_elites_evaluation.fit(phenotype_key="phenotype", data=X, elites_path=elites_path)
