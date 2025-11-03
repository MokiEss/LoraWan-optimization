#This file is to be run after running the optimization part



import pandas as pd
import ast
import csv

# Input file with the text you showed
input_file = "results/results_summary.csv"
output_file = "final_results.csv"

rows = []
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Convert string dict to real dict
        data = ast.literal_eval(line.strip('"'))
        rows.append(data)

# Extract headers automatically

headers = rows[0].keys()

# Write to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


df = pd.read_csv(output_file)


# Define your custom algorithm order
algorithm_order = ["GA", "DE", "SHADE", "CS", "PSO"]

# Apply the custom order to the 'algorithm' column
df["algorithm"] = pd.Categorical(df["algorithm"], categories=algorithm_order, ordered=True)
df = df.sort_values(by=["algorithm", "scenario_index"]).reset_index(drop=True)


if(df.iloc[0]["worst_solution"]>df.iloc[0]["best_solution"]):
    print("swapping best and worst solution columns because it is a maximization problem")
    df["worst_solution"],df["best_solution"] = df["best_solution"],df["worst_solution"]

df.to_csv(output_file, index=False)
mean_of_means = df.groupby(["algorithm", "nDevices"], observed=True)["mean_solution"].mean()
adr_df = pd.read_csv("adr_results.csv")

mean_of_adr_per_nDevices = adr_df.groupby(["algorithm","nDevices"])["pdr"].mean()

# Prepare mean_of_means DataFrame
mean_of_means_df = mean_of_means.reset_index()
mean_of_means_df = mean_of_means_df.rename(columns={"mean_solution": "mean_value"})

# Prepare mean_of_adr_per_nDevices DataFrame
mean_of_adr_df = mean_of_adr_per_nDevices.reset_index()
mean_of_adr_df = mean_of_adr_df.rename(columns={"pdr": "mean_value"})

# Combine vertically
combined_df = pd.concat([mean_of_means_df, mean_of_adr_df], ignore_index=True)

# Optional: sort by algorithm and nDevices
combined_df = combined_df.sort_values(by=["algorithm", "nDevices"]).reset_index(drop=True)

# Save to CSV
combined_df.to_csv("global_results.csv", index=False)


