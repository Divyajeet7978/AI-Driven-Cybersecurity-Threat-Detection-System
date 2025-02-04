import os
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

def identify_and_convert_types(ddf):
    for col in ddf.columns:
        try:
            # Convert columns to numeric where possible
            ddf[col] = dd.to_numeric(ddf[col], errors='coerce')
        except:
            ddf[col] = ddf[col].astype(str)
    return ddf

def process_file(file):
    try:
        # Read all columns as strings initially in smaller blocks
        ddf = dd.read_csv(file, dtype=str, assume_missing=True, low_memory=False, blocksize="32MB")
        print(f"Columns in {file}: {ddf.columns}")
        # Convert to appropriate types
        ddf = identify_and_convert_types(ddf)
        
        # Compute in chunks and write intermediate results to disk
        output_intermediate_path = f"D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/Intermediate/processed_{os.path.basename(file)}"
        os.makedirs(os.path.dirname(output_intermediate_path), exist_ok=True)
        ddf.to_csv(output_intermediate_path, index=False)
        print(f"Successfully processed and saved intermediate results for {file}")
    except PermissionError as e:
        print(f"Permission error while processing {file}: {e}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

if __name__ == '__main__':
    # Set up Dask client with specified number of workers and increased memory limit
    client = Client(n_workers=2, memory_limit='8GB')  # Adjust memory limit as needed
    print(client)

    # Path to the directory containing your original CSV files
    original_directory = 'D:/Project/Datasets'
    csv_files = [os.path.join(original_directory, f) for f in os.listdir(original_directory) if f.endswith('.csv')]

    # Process files in smaller chunks
    for file in csv_files:
        print(f"Processing file: {file}")
        process_file(file)

    # Combine intermediate results
    intermediate_dir = 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/Intermediate'
    intermediate_files = [os.path.join(intermediate_dir, f) for f in os.listdir(intermediate_dir) if f.endswith('.csv')]
    if not intermediate_files:
        print("No intermediate files to concatenate.")
    else:
        try:
            combined_ddf = dd.read_csv(intermediate_files, dtype=str, assume_missing=True, low_memory=False)

            # Compute to a pandas DataFrame
            combined_df = combined_ddf.compute()
            print("Combined DataFrame shape:", combined_df.shape)

            # Inspect missing values
            missing_values_percent = combined_df.isnull().mean()
            print("Missing values percentage in each column:\n", missing_values_percent)

            # Drop columns with more than 80% missing values
            threshold = 0.8
            columns_to_drop = combined_df.columns[missing_values_percent > threshold]
            print(f"Columns to drop (missing > {threshold * 100}%):", columns_to_drop)
            combined_df = combined_df.drop(columns=columns_to_drop)

            # Separate numeric and non-numeric columns
            numeric_columns = combined_df.select_dtypes(include=['number']).columns
            non_numeric_columns = combined_df.select_dtypes(exclude=['number']).columns

            # Convert numeric columns to float to avoid type conflicts
            combined_df[numeric_columns] = combined_df[numeric_columns].astype(float)

            # Fill missing values in numeric columns with mean
            combined_df[numeric_columns] = combined_df[numeric_columns].fillna(combined_df[numeric_columns].mean())

            # Optionally, fill missing values in non-numeric columns with a placeholder or mode
            combined_df[non_numeric_columns] = combined_df[non_numeric_columns].fillna('Unknown')

            print("Processed DataFrame shape:", combined_df.shape)

            # Save the processed DataFrame to a CSV file
            output_path = 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/ProcessedData/processed_data.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            combined_df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")

            # Continue with the rest of the pipeline (encoding, splitting, etc.)
            # Verify if 'target' column exists or identify the correct target column
            target_column = 'label'  # Assuming 'label' is the correct target column
            if target_column not in combined_df.columns:
                print(f"Target column '{target_column}' not found. Please check the dataset.")
            else:
                X = combined_df.drop(target_column, axis=1)
                y = combined_df[target_column]

                # Standardize the Data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Split the Data into Training and Testing Sets
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Train an Isolation Forest Model
                model = IsolationForest(random_state=42)
                model.fit(X_train)

                # Predict anomalies
                y_pred = model.predict(X_test)

                # Evaluate the Model
                print("Classification Report:\n", classification_report(y_test, y_pred))
                print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        except PermissionError as e:
            print(f"Permission error while reading intermediate files: {e}")
        except Exception as e:
            print(f"Error reading intermediate files: {e}")
