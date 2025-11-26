import pandas as pd

df_xgbost = pd.read_csv('xgboost_global_results.csv')
df_random_forrest = pd.read_csv('random_forest_global_results.csv')

df_xgbost_result = df_xgbost.iloc[:, [7,9]]
df_random_forrest_result = df_random_forrest.iloc[:, [7,9]]



# the mean of each number of devices of xgboost results
df_xgbost_result_mean = df_xgbost_result.groupby(df_xgbost_result.index // 20).mean()
df_xgbost_result_mean.index = [100,200,300,400,500]


# the mean of each number of devices of random forests results
df_random_forrest_result_mean = df_random_forrest_result.groupby(df_random_forrest_result.index // 20).mean()
df_random_forrest_result_mean.index = [100,200,300,400,500]


# Combine them side by side (as two columns)
combined_df = pd.concat([df_xgbost_result_mean, df_random_forrest_result_mean], axis=1)

# Optionally rename columns for clarity
combined_df.columns = ['XGBoost_Result', 'time', 'RandomForest_Result','time']

print(combined_df)
