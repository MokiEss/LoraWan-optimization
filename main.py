import surrogateGA as sga
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
df = pd.read_csv("scenarios.csv")
df = df[df["algorithm"]=="GA"]

scenario = df.iloc[0]
all_results = np.zeros((100, 9))
all_times = np.zeros((100, 1))


def run_scenario_and_runs(scenario, scenario_idx):
    """Run all 5 runs for a given scenario"""
    results_per_scenario = []
    time_per_scenario = []
    print("scenario: ", scenario)
    for run in range(5):
        bestFitness, ct = sga.runGA(scenario)
        results_per_scenario.append(bestFitness)
        time_per_scenario.append(ct)

    # Calculate statistics
    best = np.max(results_per_scenario)  # Use np.min for minimization
    worst = np.min(results_per_scenario)  # Use np.max for minimization
    mean = np.mean(results_per_scenario)
    std = np.std(results_per_scenario)
    avg_time = np.mean(time_per_scenario)

    # Store in results: [Run1, Run2, Run3, Run4, Run5, Best, Worst, Mean, Std]
    all_results[scenario_idx, :9] = results_per_scenario + [best, worst, mean, std]
    all_times[scenario_idx, 0] = avg_time


# Run everything in parallel
all_outputs = Parallel(n_jobs=-1)(
    delayed(run_scenario_and_runs)(scenario, idx)
    for idx, scenario in enumerate(df)
)

# Save with updated column names
columns = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Best', 'Worst', 'Mean', 'Std', 'Time']
combined = np.hstack([all_results, all_times])
df_output = pd.DataFrame(combined, columns=columns)
df_output.to_csv('surrogate_global_results.csv', index=False)

# Collect results



