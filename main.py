import surrogateGA as sga
import surrogateRFGA as rfsga
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

df = pd.read_csv("scenarios.csv")
df = df[df["algorithm"] == "GA"]

all_results = np.zeros((len(df), 9))
all_times = np.zeros((len(df), 1))


def run_scenario_and_runs(scenario, scenario_idx):
    """Run all 5 runs for a given scenario"""
    results_per_scenario = []
    time_per_scenario = []
    print("scenario: ", scenario_idx)

    for run in range(5):
        bestFitness, ct = sga.runGA(scenario)
        results_per_scenario.append(bestFitness)
        time_per_scenario.append(ct)

    # Calculate statistics
    best = np.max(results_per_scenario)
    worst = np.min(results_per_scenario)
    mean = np.mean(results_per_scenario)
    std = np.std(results_per_scenario)
    avg_time = np.mean(time_per_scenario)

    print("scenario: ", scenario_idx, "finished")

    # RETURN the results instead of modifying global arrays
    return scenario_idx, results_per_scenario + [best, worst, mean, std], avg_time


# Run everything in parallel
all_outputs = Parallel(n_jobs=-1)(
    delayed(run_scenario_and_runs)(row.to_dict(), idx)
    for idx, row in df.iterrows()
)

# Collect results from the parallel execution
for scenario_idx, results, avg_time in all_outputs:
    all_results[scenario_idx, :] = results
    all_times[scenario_idx, 0] = avg_time

# Save with updated column names
columns = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Best', 'Worst', 'Mean', 'Std', 'Time']
combined = np.hstack([all_results, all_times])
df_output = pd.DataFrame(combined, columns=columns)
df_output.to_csv('xgboost_global_results.csv', index=False)


print("---------------------------------------")
print("xgboost finished")
print("---------------------------------------")
# Collect results from random forest


all_results = np.zeros((len(df), 9))
all_times = np.zeros((len(df), 1))
def run_scenario_and_runs2(scenario, scenario_idx):
    """Run all 5 runs for a given scenario"""
    results_per_scenario = []
    time_per_scenario = []
    print("scenario: ", scenario_idx)

    for run in range(5):
        bestFitness, ct = rfsga.runGA(scenario)
        results_per_scenario.append(bestFitness)
        time_per_scenario.append(ct)

    # Calculate statistics
    best = np.max(results_per_scenario)
    worst = np.min(results_per_scenario)
    mean = np.mean(results_per_scenario)
    std = np.std(results_per_scenario)
    avg_time = np.mean(time_per_scenario)

    print("scenario: ", scenario_idx, "finished")

    # RETURN the results instead of modifying global arrays
    return scenario_idx, results_per_scenario + [best, worst, mean, std], avg_time


# Run everything in parallel
all_outputs = Parallel(n_jobs=-1)(
    delayed(run_scenario_and_runs2)(row.to_dict(), idx)
    for idx, row in df.iterrows()
)

# Collect results from the parallel execution
for scenario_idx, results, avg_time in all_outputs:
    all_results[scenario_idx, :] = results
    all_times[scenario_idx, 0] = avg_time

# Save with updated column names
columns = ['Run1', 'Run2', 'Run3', 'Run4', 'Run5', 'Best', 'Worst', 'Mean', 'Std', 'Time']
combined = np.hstack([all_results, all_times])
df_output = pd.DataFrame(combined, columns=columns)
df_output.to_csv('random_forest_global_results.csv', index=False)



