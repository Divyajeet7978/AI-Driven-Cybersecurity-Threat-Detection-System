import os
import pandas as pd

# Path to the directory containing your CSV files
path_to_directory = 'D:/Project/Datasets'
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

# Further processing, if needed (e.g., handling missing values, saving to CSV)
# Example: Inspect Missing Values
missing_values = combined_df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Example: Save the combined DataFrame to a CSV file
processed_output_path = 'D:/Project/ProcessedData/combined_data.csv'

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

combined_df.to_csv(processed_output_path, index=False)
print(f"Processed data saved to {processed_output_path}")

# Git commit reminders
print("Reminder: After verifying the processed data, don't forget to commit your changes!")
print("Steps to commit and push changes:")
print("1. git add .")
print("2. git commit -m 'Processed and combined CSV data'")
print("3. git push origin main")
