scenarioN = 0
scenario = None


import pandas as pd
import os
from metaheuristics import choose_metaheuristic
# Read scenarios
df = pd.read_csv("scenarios.csv", sep=",")
import concurrent.futures


# Results folder
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)
csv_file = os.path.join(results_folder, "results_summary.csv")


def run_scenario(index, row, results_folder):
    mean_curve_file = os.path.join(results_folder, f"scenario{index+1}_all_mean_convergence_curves.txt")
    with open(mean_curve_file, "w") as curve_f:
        scenario = row
        print("scenario running : ", scenario["algorithm"], scenario["nDevices"], scenario["nGateways"],
              scenario["appPeriod"], scenario["payloadSize"])
        result = choose_metaheuristic(index, scenario, curve_f)  # return dict instead of appending
        print("scenario finished : ", scenario["algorithm"], scenario["nDevices"], scenario["nGateways"],
              scenario["appPeriod"], scenario["payloadSize"])
    return result

all_results = []
with concurrent.futures.ProcessPoolExecutor() as executor:

    futures = [
        executor.submit(run_scenario, index, row, results_folder)
        for index, row in df.iterrows()
    ]
    for future in concurrent.futures.as_completed(futures):
        all_results.append(future.result())  # collect results sequentially


# Save all summary results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(csv_file, index=False)
print(f"All scenario summaries saved in {csv_file}")
print(f"All mean convergence curves saved in results")
