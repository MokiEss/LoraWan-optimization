scenarioN = 0
scenario = None


import pandas as pd
import os
from metaheuristics import choose_metaheuristic
# Read scenarios
df = pd.read_csv("scenarios.csv", sep=",")



# Results folder
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)
csv_file = os.path.join(results_folder, "results_summary.csv")
mean_curve_file = os.path.join(results_folder, "all_mean_convergence_curves.txt")

# Prepare list to store summary results
all_results = []

with open(mean_curve_file, "w") as curve_f:
    for index, row in df.iterrows():
        algorithm = row["algorithm"]
        nDevices = row["nDevices"]
        nGateways = row["nGateways"]
        radius = row["radius"]
        simulationTime = row["simulationTime"]
        appPeriod = row["appPeriod"]
        payloadSize = row["payloadSize"]
        scenario = row
        print("scenario running : ", scenario["algorithm"], scenario["nDevices"], scenario["nGateways"],scenario["appPeriod"], scenario["payloadSize"])
        all_results = choose_metaheuristic(index,scenario,all_results,curve_f)
        print("scenario finished : ", scenario["algorithm"], scenario["nDevices"], scenario["nGateways"],
              scenario["appPeriod"], scenario["payloadSize"])


# Save all summary results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(csv_file, index=False)
print(f"All scenario summaries saved in {csv_file}")
print(f"All mean convergence curves saved in {mean_curve_file}")
