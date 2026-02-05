## Credit Card Fraud Detection ML Project

An end-to-end Machine Learning pipeline designed to detect fraudulent transactions. This project utilizes supervised learning to handle highly imbalanced datasets.

##  Project Structure
- `src/`: Source code for data processing and modeling.
- `configs/`: Configuration files for model parameters.
- `data/`: Folder for datasets (ignored by Git).
- `models/`: Saved model artifacts.

##  Dataset
**Note:** The raw data is not included in this repository due to size. 
1. Download the dataset from [Kaggle: Credit Card Fraud DataSet](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the `creditcard.csv` file into the `data/raw/` directory.

##  Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it and install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`