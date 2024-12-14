import os
import pandas as pd

path_to_directory = 'D:/Project/Datasets'
csv_files = [f for f in os.listdir(path_to_directory) if f.endswith('.csv')]

# Define chunk size
chunk_size = 10000

# Initialize an empty list to hold DataFrames
chunks = []

# Read each file in chunks and append to the list
for file in csv_files:
    chunk_iter = pd.read_csv(os.path.join(path_to_directory, file), encoding='ISO-8859-1', chunksize=chunk_size, low_memory=False)
    for chunk in chunk_iter:
        chunks.append(chunk)

# Combine all chunks into a single DataFrame
if chunks:
    combined_df = pd.concat(chunks, ignore_index=True)
else:
    combined_df = pd.DataFrame()  # Create an empty DataFrame if no chunks were read

# Verify the combined DataFrame
print(combined_df.head())
