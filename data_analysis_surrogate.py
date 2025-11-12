import pandas as pd

df_xgbost = pd.read_csv('surrogate_global_results.csv')
df_random_forrest = pd.read_csv('random_forest_global_results.csv')

df_xgbost_result = df_xgbost.iloc[:, [7,9]]
df_random_forrest_result = df_random_forrest.iloc[:, [7,9]]

# Combine them side by side (as two columns)
combined_df = pd.concat([df_xgbost_result, df_random_forrest_result], axis=1)

# Optionally rename columns for clarity
combined_df.columns = ['XGBoost_Result', 'time', 'RandomForest_Result','time']

print(combined_df)
