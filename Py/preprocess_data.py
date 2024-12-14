import os
import pandas as pd

# Update the path to the directory containing your CSV files
path_to_directory = 'D:/Datasets'  # Replace with the actual path
csv_files = [f for f in os.listdir(path_to_directory) if f.endswith('.csv')]

# Define chunk size
chunk_size = 10000

# Initialize an empty list to hold DataFrames
chunks = []

# Read each file in chunks and append to the list
for file in csv_files:
    file_path = os.path.join(path_to_directory, file)
    try:
        chunk_iter = pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=chunk_size, low_memory=False)
        for chunk in chunk_iter:
            chunks.append(chunk)
        print(f"Successfully processed {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all chunks into a single DataFrame
if chunks:
    combined_df = pd.concat(chunks, ignore_index=True)
else:
    combined_df = pd.DataFrame()  # Create an empty DataFrame if no chunks were read

# Verify the combined DataFrame
print("Combined DataFrame head:\n", combined_df.head())
print("Combined DataFrame shape:", combined_df.shape)

# Inspect Missing Values
missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Drop Columns with a High Percentage of Missing Values
threshold = 0.5  # 50%
combined_df = combined_df.loc[:, combined_df.isnull().mean() < threshold]

# Fill Remaining Missing Values with Mean
combined_df = combined_df.fillna(combined_df.mean())

# Print to verify the missing values are handled
print("Processed DataFrame head:\n", combined_df.head())
print("Processed DataFrame shape:", combined_df.shape)

# Save the processed DataFrame to a CSV file in a non-ignored directory
processed_output_path = 'D:/Project/ProcessedData/processed_data.csv'
combined_df.to_csv(processed_output_path, index=False)
print(f"Processed data saved to {processed_output_path}")
