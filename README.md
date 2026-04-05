Customer Segmentation using Machine Learning

A production-style machine learning project that segments customers based on purchasing behavior using RFM analysis and KMeans clustering, with a Flask web app for real-time predictions.

 Overview

Businesses have different types of customers, but treating them the same reduces effectiveness.
This project builds an intelligent system to:

Analyze customer transaction data
Segment customers into meaningful groups
Enable data-driven marketing strategies
 Key Features
 RFM-based feature engineering (Recency, Frequency, Monetary)
 Product-level behavior analysis using top purchased items
 Data preprocessing with scaling and cleaning
 Dimensionality reduction using PCA
 Customer segmentation using KMeans clustering
 Flask web app for CSV upload and real-time predictions
 Model persistence using Joblib
 Machine Learning Pipeline
1. Data Processing
Removed missing values
Converted date fields
Created sales features
2. Feature Engineering
RFM metrics:
Recency (last purchase)
Frequency (number of purchases)
Monetary (total spending)
Top 20 product features using pivot tables
3. Preprocessing
Feature scaling using StandardScaler
Dimensionality reduction using PCA
4. Modeling
Applied KMeans clustering to segment customers
Assigned each customer to a cluster
 Output
Customer segments (clusters) with behavioral insights
Segment-wise statistics (average recency, frequency, monetary)
Visual distribution of customer groups
Downloadable segmented dataset
 Web Application
Upload CSV file
Automatically generate customer segments
View cluster insights and charts
Download processed results
 Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-learn
ML Models: KMeans, PCA, StandardScaler
Backend: Flask
Frontend: HTML, CSS
Model Storage: Joblib
Config Management: YAML
 Project Structure
├── app.py                  # Flask web app
├── main.py                 # CLI entry point
├── config/
│   └── config.yaml         # Configuration file
├── src/
│   ├── data_processing/    # Data cleaning & feature engineering
│   ├── models/             # Model training logic
│   ├── pipelines/          # Training & inference pipelines
│   └── utils/              # Logging
├── models/                 # Saved ML artifacts
├── data/                   # Raw & processed data
├── templates/              # HTML files
├── static/                 # CSS
└── README.md
