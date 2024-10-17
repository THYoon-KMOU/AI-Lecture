import pandas as pd
import glob
import os

# Specify the folder path
folder_path = 'dataset/fff'

# List to store dataframes
dfs = []

# Get all csv files in the directory
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Loop through all csv files
for filename in csv_files:
    # Read the csv file
    df = pd.read_csv(filename)
    # Select only the columns you are interested in
    df = df[['brake_pressure', 'long_accel', 'steering_angle', 'wheel_speed']]
    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes in the list
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new csv file
combined_df.to_csv('combined_traindata.csv', index=False)
