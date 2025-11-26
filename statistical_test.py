import os
import re
import numpy as np
from scipy.stats import wilcoxon
# -------------------- YOU MUST FILL THESE --------------------
folder_path = "results"  # <--- CHANGE
ALGORITHMS = ["GA", "DE", "SHADE", "PSO", "CS"]  # <--- CHANGE
NUM_SCENARIOS = 100
RUNS_PER_ALG = 5
# ----------- Load all files in sorted order -----------
# ----------- Helper to extract scenario number -----------
def get_scenario(filename):
    m = re.search(r"scenario(\d+)", filename)
    return int(m.group(1)) if m else None

# ----------- Collect all files with scenario info -----------
files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

# Extract scenario numbers
files_info = []
for f in files:
    scenario = get_scenario(f)
    if scenario is not None:
        files_info.append((scenario, f))

# Sort files by scenario number
files_info.sort(key=lambda x: x[0])  # ascending scenario

sorted_files = [f for _, f in files_info]

# ----------- Prepare final array -----------
data = np.zeros((NUM_SCENARIOS * RUNS_PER_ALG, len(ALGORITHMS)))

# ----------- Fill array algorithm by algorithm -----------
for alg_idx, alg in enumerate(ALGORITHMS):
    # slice 100 files for this algorithm (files are already sorted by scenario)
    start_file = alg_idx * NUM_SCENARIOS
    end_file = (alg_idx + 1) * NUM_SCENARIOS
    alg_files = sorted_files[start_file:end_file]

    for scenario_idx, filename in enumerate(alg_files):
        # read 5 runs
        runs = []
        with open(os.path.join(folder_path, filename), "r") as f:
            read = False
            for line in f:
                if "solutions runs are" in line:
                    read = True
                    continue
                if read:
                    line = line.strip()
                    if line == "":
                        continue
                    try:
                        runs.append(float(line))
                    except:
                        break
        if len(runs) != RUNS_PER_ALG:
            print(f" WARNING: {filename} does not contain {RUNS_PER_ALG} runs!")
            continue

        # place runs into array
        start_row = scenario_idx * RUNS_PER_ALG
        data[start_row:start_row+RUNS_PER_ALG, alg_idx] = runs

# ----------- Done -----------

# Flatten runs per algorithm

runs_per_algorithm = [data[:, i] for i in range(len(ALGORITHMS))]
print(len(runs_per_algorithm))
# statistical test
stat, p = wilcoxon(runs_per_algorithm[1], runs_per_algorithm[2])
print(f"GA vs DE (all runs): statistic={stat}, p-value={p}")