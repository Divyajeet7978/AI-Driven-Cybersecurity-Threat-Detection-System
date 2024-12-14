import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import logging

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Starting data processing')

# Path to the directory containing your CSV files
path_to_directory = 'D:/Project/Datasets'  # Replace with the actual path
csv_files = [f for f in os.listdir(path_to_directory) if f.endswith('.csv')]

# Print the list of CSV files
print("CSV Files:", csv_files)

# Define chunk size
chunk_size = 10000

# Initialize an empty list to hold DataFrames
chunks = []

# Read each file in chunks and append to the list
for file in csv_files:
    file_path = os.path.join(path_to_directory, file)
    print(f"Processing file: {file_path}")
    try:
        chunk_iter = pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=chunk_size, low_memory=False)
        for chunk in chunk_iter:
            print(f"Appending chunk of size: {chunk.shape}")
            chunks.append(chunk)
        logging.info(f"Successfully processed {file}")
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")

# Combine all chunks into a single DataFrame
if chunks:
    combined_df = pd.concat(chunks, ignore_index=True)
else:
    combined_df = pd.DataFrame()  # Create an empty DataFrame if no chunks were read

# Verify the combined DataFrame
logging.info(f"Combined DataFrame head:\n{combined_df.head()}")
logging.info(f"Combined DataFrame shape: {combined_df.shape}")

# Inspect Missing Values
missing_values = combined_df.isnull().sum()
logging.info(f"Missing values in each column:\n{missing_values}")

# Drop Columns with a High Percentage of Missing Values
threshold = 0.5  # 50%
combined_df = combined_df.loc[:, combined_df.isnull().mean() < threshold]

# Fill Remaining Missing Values with Mean
combined_df = combined_df.fillna(combined_df.mean())

# Print to verify the missing values are handled
logging.info(f"Processed DataFrame head:\n{combined_df.head()}")
logging.info(f"Processed DataFrame shape: {combined_df.shape}")

# Save the processed DataFrame to a CSV file in a non-ignored directory
processed_output_path = 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/ProcessedData/processed_data.csv'

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

combined_df.to_csv(processed_output_path, index=False)
logging.info(f"Processed data saved to {processed_output_path}")

logging.info('Finished data processing, starting model training')

# Check the columns to ensure 'label' is correct
print("Columns in combined_df:", combined_df.columns)

# Split the Data
X = combined_df.drop('label', axis=1)  # Replace 'label' with your target column name
y = combined_df['label']  # Replace 'label' with your target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an Isolation Forest Model
model = IsolationForest(n_estimators=100, random_state=42)
model.fit(X_train)

logging.info('Model training completed')

# Predict anomalies
y_pred = model.predict(X_test)
y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to 1 for anomalies and 0 for normal

# Evaluate the Model
classification_report_output = classification_report(y_test, y_pred)
confusion_matrix_output = confusion_matrix(y_test, y_pred)
logging.info('Model evaluation completed')
logging.info(f'Classification Report:\n{classification_report_output}')
logging.info(f'Confusion Matrix:\n{confusion_matrix_output}')

# Save the trained model
model_output_path = 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/Models/isolation_forest_model.joblib'
dump(model, model_output_path)
logging.info(f'Model saved to {model_output_path}')

# Save the evaluation metrics to a file
evaluation_output_path = 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/Models/evaluation_report.txt'
with open(evaluation_output_path, 'w') as f:
    f.write(f'Classification Report:\n{classification_report_output}\n')
    f.write(f'Confusion Matrix:\n{confusion_matrix_output}\n')
logging.info(f'Evaluation report saved to {evaluation_output_path}')
