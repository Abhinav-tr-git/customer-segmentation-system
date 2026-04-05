# How to Run

Follow these steps to run the Customer Segmentation Production System.

## 1. Setup Environment
Ensure you have Python installed, then install the required dependencies:
```powershell
pip install -r requirements.txt
```

## 2. Project Organization
The project expects the following structure:
- `data/raw/`: Place your transaction data here (e.g., `int_online_tx.csv`).
- `config/config.yaml`: Tune model parameters and data paths.

## 3. Operations

### Train the Model
This will clean the data, engineer features (RFM + Item-level), and train the clustering model (PCA + KMeans).
```powershell
python main.py --mode train
```
**Output**: 
- Model artifacts in `models/`.
- Segmented customer data in `data/processed/customer_segments.csv`.

### Run Inference
Assign clusters to a new dataset using the previously trained and saved model.
```powershell
python main.py --mode infer --input data/raw/int_online_tx.csv
```

## 4. Troubleshooting & Logs
Check the `logs/pipeline.log` file for detailed execution traces and any issues encountered during the run.
