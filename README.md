## Credit Card Fraud Detection ML Project

An end-to-end Machine Learning pipeline designed to detect fraudulent transactions. This project utilizes supervised learning to handle highly imbalanced datasets.

##  Project Structure
- `src/`: Source code for data processing and modeling.
- `configs/`: Configuration files for model parameters.
- `data/`: Folder for datasets (ignored by Git).
- `models/`: Saved model artifacts.

##  Dataset 
1. Download the dataset from [Kaggle: credit card fraud dataset](https://www.kaggle.com/datasets/bhadramohit/credit-card-fraud-detection/data).
2. Place the `credit_card_fraud_dataset.csv` file into the `data/raw/` directory.

##  Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it and install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
