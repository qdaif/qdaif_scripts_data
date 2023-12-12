### Data
- Download from the [Google Drive folder](https://drive.google.com/drive/folders/1MtFpkf2PPOws0DQmOvKnrulhHS2r4Jqx?usp=sharing), and save contents into data directory in local repo base directory
- `data/histories_opinions_stories` for raw generated text samples and results in Opinions and Stories domains
    - Organized into `baselines`, `qdaif`, and `qdef` (embedding feedback experiments)
    - Each domain subfolder contains the results from each method, containing `history.jsonl` files with generated solutions, and fitness (quality measure) and phenotype (diversity measure)
    - Additional diversity-seeking baseline results are in their own folder (LMX, ROUGE-L and LMX, NSAIF, with quality AI feedback filters)
    - Folders for 5 repeated (rng) runs
    - Default method for QDAIF is `lmx_near_seeded_init`
- `data/histories_poetry` for results in the Poetry domain
    - GPT-4 and GPT-3.5 Turbo denote models used for LMX mutation
    - `multi_component_quality` was tested but not fully explored -- featured in the poetry chain figure in the paper
- `data/histories_code` contains both histories, and results of running HumanEval (no. 88) on parsed generated code

### To get the performance bars
- Sheets saved in `perf_sheets` for plotting
- First load history.jsonl files into get_elites.py to obtain elites.py
- Then run get_csv_stats.py to get the results spreadsheets
- Format sheets and run perf_csv_compare.ipynb to plot

### Other scripts
- Barplot script is for the (uniform vs. non-uniform) bin distribution analysis
- Baseline method scripts cover both the 4 original baselines, and extended diversity-seeking baselines
- `evaluate_qd_jsonl_batch.py` is for computing phenotypes for baseline outputs in history.jsonl files, where baselines don't record diversity measures usually
- `get_qd_score.ipynb` is for line plots
- `map_elites.py` contains classes for setting up archives just for evaluating results
- poetry scripts are separate