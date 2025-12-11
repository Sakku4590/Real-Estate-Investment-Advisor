import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging
from dvclive import Live

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise
    
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a given model and return metrics."""
    try:
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise
    
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise
    
def main():
    try:
        # Load test data
        test_data = load_data('./data/final/test_encoded.csv')
        X_test = test_data.drop(["Good_Investment"], axis=1)
        y_test = test_data["Good_Investment"]

        # Load ALL models
        log_model = load_model('./models/LogisticRegression.pkl')
        rf_model  = load_model('./models/RandomForestClassifier.pkl')
        xgb_model = load_model('./models/XgboostClassifier.pkl')
        # Evaluate each model
        log_metrics = evaluate_model(log_model, X_test, y_test)
        rf_metrics  = evaluate_model(rf_model, X_test, y_test)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        
        # Log metrics using DVC Live
        with Live(save_dvc_exp=True) as live:

            # Logistic Regression
            live.log_metric("log_accuracy",  log_metrics["accuracy"])
            live.log_metric("log_precision", log_metrics["precision"])
            live.log_metric("log_recall",    log_metrics["recall"])
            live.log_metric("log_roc_auc",   log_metrics["roc_auc"])

            # Random Forest
            live.log_metric("rf_accuracy",  rf_metrics["accuracy"])
            live.log_metric("rf_precision", rf_metrics["precision"])
            live.log_metric("rf_recall",    rf_metrics["recall"])
            live.log_metric("rf_roc_auc",   rf_metrics["roc_auc"])

            # XGBoost
            live.log_metric("xgb_accuracy",  xgb_metrics["accuracy"])
            live.log_metric("xgb_precision", xgb_metrics["precision"])
            live.log_metric("xgb_recall",    xgb_metrics["recall"])
            live.log_metric("xgb_roc_auc",   xgb_metrics["roc_auc"])
            
        # Save all metrics
        save_metrics({
            "logistic_regression": log_metrics,
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics
        }, "reports/metrics.json")

    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()