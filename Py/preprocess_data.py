import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Path to the directory containing your CSV files
path_to_directory = 'D:/Project/Datasets'  # Replace with the actual path
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
processed_output_path = 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/ProcessedData/processed_data.csv'

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

combined_df.to_csv(processed_output_path, index=False)
print(f"Processed data saved to {processed_output_path}")

# Split the Data
X = combined_df.drop('label', axis=1)  # Replace 'label' with your target column name
y = combined_df['label']  # Replace 'label' with your target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an Isolation Forest Model
model = IsolationForest(n_estimators=100, random_state=42)
model.fit(X_train)

# Predict anomalies
y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to 1 for anomalies and 0 for normal

# Evaluate the Model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
