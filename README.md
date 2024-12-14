# AI-Driven Cybersecurity Threat Detection

## Project Overview
This project aims to build an AI-driven cybersecurity platform to detect and respond to threats in real-time using machine learning and anomaly detection techniques.

## Setup Instructions
1. **Install Python**:
   - Ensure Python 3.8 or higher is installed.
   - Verify the installation by running `python --version`.
2. **Create a Virtual Environment**:
   - Navigate to your project directory.
   - Create a virtual environment: `python -m venv venv`.
   - Activate the virtual environment:
     - On macOS/Linux: `source venv/bin/activate`
     - On Windows: `venv\Scripts\activate`
3. **Install Required Libraries**:
   - Create a `requirements.txt` file with the necessary libraries.
   - Install the libraries using `pip install -r requirements.txt`.

## Data Collection
1. **Download Datasets**:
   - UNSW-NB15 Dataset: [Download Link](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - KDD Cup 1999 Dataset: [Download Link](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
   - CICIDS 2017 Dataset: [Download Link](https://www.unb.ca/cic/datasets/ids-2017.html)
2. **Preprocess Data**:
   - Load datasets using Pandas.
   - Handle missing values.
   - Encode categorical variables.
   - Split features and target variable.
   - Standardize the data.
   - Split into training and testing sets.

## Model Training
1. **Train an Isolation Forest Model**:
   - Initialize and fit the model with training data.
   - Predict anomalies on the test data.
2. **Evaluate the Model**:
   - Generate classification reports and confusion matrices.

## Real-Time Monitoring
1. **Set Up Splunk**:
   - Download, install, and configure Splunk.
   - Add data inputs to monitor network traffic or log files.
   - Verify data collection using SPL queries.
2. **Fetch Data from Splunk Using Python**:
   - Install and use the Splunk SDK for Python to fetch and process data.
   - Prepare data for model inference.

## API Documentation
1. **Develop Flask API**:
   - Set up project structure.
   - Write Flask application code.
   - Prepare `requirements.txt`.
   - Test the API locally.
2. **API Endpoints**:
   - `/predict`: Predict anomalies based on input data.
   - Example request:
     ```json
     {
         "data": [[0.1, 0.2, 0.3, ...]]
     }
     ```
   - Example response:
     ```json
     {
         "prediction": [[0.5]]
     }
     ```

## Deployment
1. **Containerize with Docker**:
   - Create a Dockerfile and build the Docker image.
   - Run the Docker container locally and test the application.
2. **Deploy on AWS**:
   - Set up AWS account and configure IAM user.
   - Push Docker image to Amazon ECR.
   - Create ECS task definition and deploy using AWS Fargate.

# Custom License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to view the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**Restrictions**:
- The Software shall not be used for commercial purposes.
- The Software shall not be reproduced, distributed, or used to create derivative works.
- The code and related files in this repository are provided for viewing purposes only.
- No permission is granted to use, modify, or distribute the code or related files, in whole or in part, for any purpose.

© 2025 Divyajeet. All rights reserved.
