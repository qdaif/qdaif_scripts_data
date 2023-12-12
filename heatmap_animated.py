import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from map_elites import MAPElites2DEvaluation

plt.rcParams['figure.constrained_layout.use'] = True

def load_results(path):
    X = pd.read_json(path, lines=True)
    return X

# load iteration data of baseline and QDAIF, for median QD score run (out of 5 re-runs)
data_1 = load_results("data/histories_opinions_stories/qdaif/stories_genre_ending/lmx_near_seeded_init/1/history.jsonl")
data_2 = load_results("data/histories_opinions_stories/baselines/lmx_quality_only/stories_genre_ending/3/history.jsonl")

X_BINS = [0.005, 0.02, 0.05, 0.20, 0.50, 0.80, 0.95, 0.98, 0.995]
Y_BINS = [0.005, 0.02, 0.05, 0.20, 0.50, 0.80, 0.95, 0.98, 0.995]

# compute the state of the archives at each iteration
map_elites_evaluation_1 = MAPElites2DEvaluation(history_length=len(data_1), x_bins=X_BINS, y_bins=Y_BINS, start=(0,0), stop=(1,1))
map_elites_evaluation_2 = MAPElites2DEvaluation(history_length=len(data_2), x_bins=X_BINS, y_bins=Y_BINS, start=(0,0), stop=(1,1))

map_elites_evaluation_1.fit(phenotype_key="phenotype", data=data_1)
map_elites_evaluation_2.fit(phenotype_key="phenotype", data=data_2)

archive_1 = map_elites_evaluation_1.archive
archive_1[archive_1 == -np.inf] = 0

archive_2 = map_elites_evaluation_2.archive
archive_2[archive_2 == -np.inf] = 0

# initialize the figure and axes
fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig) 
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
g1 = sns.heatmap(np.flip(archive_1[0], 0), cmap=sns.color_palette("rocket", as_cmap=True), ax=ax1, vmin=0, vmax=1)
g2 = sns.heatmap(np.flip(archive_2[0], 0), cmap=sns.color_palette("rocket", as_cmap=True), ax=ax2, vmin=0, vmax=1)

cbar1 = g1.collections[0].colorbar
cbar1.ax.tick_params(labelsize=15)

cbar2 = g2.collections[0].colorbar
cbar2.ax.tick_params(labelsize=15)

def update(frame):
    ax1.clear()
    ax2.clear()

    g1 = sns.heatmap(np.flip(archive_1[frame], 0), cmap=sns.color_palette("rocket", as_cmap=True), square=True, linewidths=.5, ax=ax1, vmin=0, vmax=1, cbar = False)
    g2 = sns.heatmap(np.flip(archive_2[frame], 0), cmap=sns.color_palette("rocket", as_cmap=True), square=True, linewidths=.5, ax=ax2, vmin=0, vmax=1, cbar = False)

    # account for swapping of labels (and their corresponding range limits) from analysis scripts ran for baseline methods
    g1.invert_yaxis() # for QDAIF
    g1.set_xticks([0, 5, 10], labels=['0.0', '0.5', '1.0'], fontsize=12)
    g1.set_yticks([0, 5, 10], labels=['0.0', '0.5', '1.0'], fontsize=12) # for QDAIF
    
    g2.set_xticks([0, 5, 10], labels=['0.0', '0.5', '1.0'], fontsize=12)
    g2.set_yticks([0, 5, 10], labels=['1.0', '0.5', '0.0'], fontsize=12) # for baselines
    
    g1.set_xlabel("Ending (Tragic to Happy)", fontsize=15)
    g1.set_ylabel("Genre (Romance to Horror)", fontsize=15)
    g2.set_xlabel("Ending (Tragic to Happy)", fontsize=15)
    g2.set_ylabel("Genre (Romance to Horror)", fontsize=15)
    
    ax1.set_title("QDAIF", fontsize=40)
    ax2.set_title("Baseline", fontsize=40)

# create and save animation
ani = FuncAnimation(fig, update, frames=range(0, 2000, 10), repeat=False)

output_gif_path = "animated_heatmap_qdaif_vs_baseline.mp4"
ani.save(output_gif_path, writer='ffmpeg', fps=60)
