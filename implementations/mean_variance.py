
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('new_output.csv')

# Ensure the 'seed' column is treated as numeric and drop rows where 'seed' is not a digit
data = data[pd.to_numeric(data['seed'], errors='coerce').notnull()]

# Convert numerical columns to float if not already
numerical_cols = ['ACC_10', 'SP_10', 'EO_10', 'ACC_100', 'SP_100', 'EO_100']
data[numerical_cols] = data[numerical_cols].astype(float)

# Group by 'dataset' and 'model', then calculate the mean and variance for each group
grouped = data.groupby(['dataset', 'model'])

mean_values = grouped[numerical_cols].mean() * 100
variance_values = grouped[numerical_cols].var() * 100

# For simplification, let's just create a DataFrame with the means for now
mean_df = mean_values.reset_index()

# You can save this DataFrame to a CSV file
mean_df.to_csv('averages_by_dataset_and_model.csv', index=False)

print("Averages by dataset and model saved to averages_by_dataset_and_model.csv")

# # Extracting numerical values and grouping them by columns
# import numpy as np

# data = [
#     ["bail",1,"gat",0.9313413858868405,0.06339504613094221,0.022351293089662327,0.9277389277389277,0.05311339380610014,0.019197727897857675
# ],
# ["bail",2,"gat",0.9404534859080315,0.08132555445526057,0.027198347054734673,0.9360033905488452,0.07328898936865902,0.020039410500538812
# ],
# ["bail",3,"gat",0.955287137105319,0.06300302339493302,0.0025819672131147664,0.9571943208306846,0.05659461178207842,0.0053688524590164155
# ], 
# ["bail",4,"gat",0.9465988556897649,0.08152535813020151,0.05104166666666665,0.9436321254503073,0.07970735076545693,0.04901960784313719
# ], 
# ["bail",5,"gat",0.9578300487391397,0.08543294940353763,0.014423076923076872,0.95910150455605,0.07756581653640476,0.005805899608865683
# ]
# ]

# def calc(): 
#     column_data = list(zip(*[row[3:] for row in data]))  # Exclude non-numeric columns

#     # Calculate mean and variance for each column
#     mean_variance_per_column = [(np.mean(col) * 100, np.var(col) * 100) for col in column_data]
#     print(mean_variance_per_column)
#     return mean_variance_per_column
# # return calc()
# calc()

# import pandas as pd

# # Load the data from the CSV file
# data = pd.read_csv('new_output.csv')

# dataset_value = data['dataset'].iloc[0]
# model_value = data['model'].iloc[0]

# # Since your CSV has repeating headers, let's clean that up first
# data = data[data['seed'].apply(lambda x: str(x).isdigit())]
# # Convert numerical columns to float
# numerical_cols = ['ACC_10', 'SP_10', 'EO_10', 'ACC_100', 'SP_100', 'EO_100']
# data[numerical_cols] = data[numerical_cols].astype(float)
# # print(data[numerical_cols])
# # Calculate the mean of each column
# mean_values = data[numerical_cols].mean() * 100
# variance_values = data[numerical_cols].var() * 100

# # Print the mean of each column
# print("Mean of each column:")
# print(mean_values)

# print("Variance of each column:")
# print(variance_values)

# # Combine mean and variance into a single string per column
# combined_results = {}
# for col in numerical_cols:
#     combined_results[col] = f"{mean_values[col]:.6f} ± {variance_values[col]:.6f}"

# # Convert the combined results into a DataFrame for easy CSV export
# results_df = pd.DataFrame.from_dict(combined_results, orient='index', columns=['Mean ± Variance'])

# results_df['Dataset'] = dataset_value
# results_df['Model'] = model_value

# results_df = results_df[['Dataset', 'Model', 'Mean ± Variance']]

# # Save the results to a CSV file
# results_df.to_csv('mean_variance_results_with_info.csv', index_label='Metric')

# print("Results saved to mean_variance_results_with_info.csv")