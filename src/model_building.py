import os
import numpy as np
import pandas as pd
from typing import Tuple
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import yaml

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str) ->dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrived from %s',params_path)
        return params
    except FileExistsError:
        logger.error('File not Found %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YMAL error: %s',e)
        raise
    except Exception as e:
        raise

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s',file_path,df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s',e)
        raise
    
def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict):
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of sample in X_train and y_train must be the same.")
        
        log_model = LogisticRegression(max_iter=params['max_iter'])
        rf_model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        xgb_model = XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            eval_metric=params['eval_metric']
        )
        logger.debug('Model training started with %d samples',X_train.shape[0])
        
        log_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        
        logger.debug('Model training completed')
        return log_model,rf_model,xgb_model
    
    except ValueError as e:
        logger.error('ValueError During model trainnig: %s',e)
        raise
    except Exception as e:
        logger.error('Error during model trainning: %s',e)
        raise
    
def save_model(model,file_path:str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s',file_path)
        
    except FileNotFoundError as e:
        logger.error('File path not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s',e)
        raise
    
def main():
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/final/train_encoded.csv')

        X_train = train_data.drop(["Good_Investment"], axis=1)
        y_train = train_data["Good_Investment"]

        # Train models ONCE
        log_model, rf_model, xgb_model = train_model(X_train, y_train,params)

        # Save paths
        Log_model_save_path = 'models/LogisticRegression.pkl'
        Reg_model_save_path = 'models/RandomForestClassifier.pkl'
        Xgb_model_save_path = 'models/XgboostClassifier.pkl'

        # Save models separately
        save_model(log_model, Log_model_save_path)
        save_model(rf_model, Reg_model_save_path)
        save_model(xgb_model, Xgb_model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        
if __name__=='__main__':
    main()
