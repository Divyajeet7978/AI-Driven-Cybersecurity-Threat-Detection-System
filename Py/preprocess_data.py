import os
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE
from joblib import dump, load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data loading, cleaning and preprocessing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.numeric_columns = None
        self.categorical_columns = None
        self.feature_columns = None
        self.target_column = None
        
    def _detect_column_types(self, ddf: dd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical columns"""
        # Sample a portion of data for type detection
        sample_df = ddf.head(n=1000)
        
        numeric_cols = []
        categorical_cols = []
        
        for col in sample_df.columns:
            # Skip target column if specified
            if self.config['target_column'] and col == self.config['target_column']:
                continue
                
            # Check if numeric
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                numeric_cols.append(col)
            else:
                # Check if it's actually a string representation of numbers
                try:
                    pd.to_numeric(sample_df[col])
                    numeric_cols.append(col)
                except ValueError:
                    # Check cardinality for categoricals
                    unique_count = sample_df[col].nunique()
                    if 1 < unique_count <= self.config['max_categories']:
                        categorical_cols.append(col)
                    else:
                        # High cardinality - might want to drop or hash
                        if self.config['handle_high_cardinality'] == 'drop':
                            logger.warning(f"Dropping high cardinality column: {col}")
                            continue
                        
        return numeric_cols, categorical_cols
    
    def load_and_concatenate(self, file_paths: List[str]) -> dd.DataFrame:
        """Load and concatenate multiple CSV files"""
        try:
            # Read with optimized chunks and only necessary columns if specified
            dfs = []
            for file in file_paths:
                ddf = dd.read_csv(
                    file,
                    assume_missing=True,
                    blocksize=self.config['blocksize'],
                    dtype='object',  # Read as string initially for type inference
                    usecols=self.config.get('columns_to_use')
                )
                dfs.append(ddf)
            
            combined_ddf = dd.concat(dfs, axis=0)
            return combined_ddf
        except Exception as e:
            logger.error(f"Error loading files: {e}")
            raise
    
    def clean_data(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Perform data cleaning operations"""
        # Drop high missing value columns
        missing_percent = ddf.isnull().mean().compute()
        cols_to_drop = missing_percent[missing_percent > self.config['missing_threshold']].index.tolist()
        ddf = ddf.drop(columns=cols_to_drop)
        
        # Convert types
        numeric_cols, categorical_cols = self._detect_column_types(ddf)
        
        # Convert numeric columns
        for col in numeric_cols:
            ddf[col] = dd.to_numeric(ddf[col], errors='coerce')
        
        # Convert categorical columns
        for col in categorical_cols:
            ddf[col] = ddf[col].astype('category')
        
        # Handle remaining object columns
        remaining_obj_cols = [
            col for col in ddf.columns 
            if col not in numeric_cols + categorical_cols 
            and pd.api.types.is_string_dtype(ddf[col])
        ]
        for col in remaining_obj_cols:
            # Try to parse dates
            try:
                ddf[col] = dd.to_datetime(ddf[col], errors='coerce')
                numeric_cols.append(col)  # Datetimes can be treated as numeric
            except:
                # If not date, drop or keep as string based on config
                if self.config['drop_unparseable_objects']:
                    ddf = ddf.drop(columns=[col])
                else:
                    ddf[col] = ddf[col].astype('category')
                    categorical_cols.append(col)
        
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        self.feature_columns = numeric_cols + categorical_cols
        
        return ddf
    
    def preprocess(self, ddf: dd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for modeling"""
        # Compute to pandas if fits in memory
        try:
            df = ddf.compute()
        except MemoryError:
            logger.error("Data too large to fit in memory. Consider sampling or additional preprocessing steps.")
            raise
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df[self.config['target_column']]
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('variance_threshold', VarianceThreshold(threshold=0.1))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_columns),
                ('cat', categorical_transformer, self.categorical_columns)
            ],
            remainder='drop'
        )
        
        # Fit and transform
        X_processed = preprocessor.fit_transform(X)
        
        return X_processed, y.values

class AnomalyDetector:
    """Handles anomaly detection model training and evaluation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.preprocessor = None
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the anomaly detection model"""
        # Handle class imbalance if needed
        if self.config['handle_imbalance'] and len(np.unique(y)) > 1:
            smote = SMOTE(random_state=self.config['random_state'])
            X, y = smote.fit_resample(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Train model
        self.model = IsolationForest(
            n_estimators=self.config['n_estimators'],
            max_samples=self.config['max_samples'],
            contamination=self.config['contamination'],
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        self.model.fit(X_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    
    def save_model(self, output_dir: str) -> None:
        """Save model and preprocessing artifacts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        dump(self.model, os.path.join(output_dir, 'isolation_forest.joblib'))
        logger.info(f"Model saved to {output_dir}")

def main():
    # Configuration - could be moved to a separate config file
    config = {
        'input_dir': 'D:/Project/Datasets',
        'intermediate_dir': 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/Intermediate',
        'output_dir': 'D:/Project/Repo/AI-Driven-Cybersecurity-Threat-Detection-System/ProcessedData',
        'target_column': 'label',
        'blocksize': '64MB',  # Adjusted based on system memory
        'missing_threshold': 0.8,
        'max_categories': 50,
        'handle_high_cardinality': 'keep',  # or 'drop'
        'drop_unparseable_objects': False,
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100,
        'max_samples': 'auto',
        'contamination': 'auto',
        'n_jobs': -1,
        'handle_imbalance': True
    }
    
    try:
        # Initialize Dask client with adaptive scaling
        cluster = LocalCluster(
            n_workers=min(4, os.cpu_count()), 
            threads_per_worker=1,
            memory_limit='auto'  # Use up to 75% of available memory
        )
        client = Client(cluster)
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        
        # Initialize processor
        processor = DataProcessor(config)
        
        # Load and process data
        input_files = [
            os.path.join(config['input_dir'], f) 
            for f in os.listdir(config['input_dir']) 
            if f.endswith('.csv')
        ]
        
        if not input_files:
            logger.error("No CSV files found in input directory")
            return
        
        logger.info(f"Processing {len(input_files)} files")
        combined_ddf = processor.load_and_concatenate(input_files)
        cleaned_ddf = processor.clean_data(combined_ddf)
        
        # Save cleaned data
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(config['output_dir'], 'processed_data.parquet')
        cleaned_ddf.to_parquet(output_path)
        logger.info(f"Cleaned data saved to {output_path}")
        
        # Preprocess for modeling
        X, y = processor.preprocess(cleaned_ddf)
        
        # Train model
        detector = AnomalyDetector(config)
        detector.train_model(X, y)
        detector.save_model(config['output_dir'])
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        client.close()
        cluster.close()

if __name__ == '__main__':
    main()
