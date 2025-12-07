import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_labelEncoder(train_data: pd.DataFrame, test_data: pd.DataFrame):
    try:
        label_col = ["Availability_Status","Furnished_Status","Public_Transport_Accessibility","Parking_Space","Security","Locality"]
        onehot_cols = ["State", "City", "Property_Type", "Facing", "Owner_Type"]
        
        le = LabelEncoder()

        for col in label_col:
            train_data[col] = le.fit_transform(train_data[col])
            test_data[col] = le.transform(test_data[col])
            
        ohe = OneHotEncoder(drop="first", sparse_output=False)    

        # TRAIN
        ohe_train = ohe.fit_transform(train_data[onehot_cols])
        ohe_df_train = pd.DataFrame(ohe_train, columns=ohe.get_feature_names_out(onehot_cols))

        # TEST (use transform, NOT fit_transform!)
        ohe_test = ohe.transform(test_data[onehot_cols])
        ohe_df_test = pd.DataFrame(ohe_test, columns=ohe.get_feature_names_out(onehot_cols))
        
        # Combine encoded with original dataset (remove original categorical columns)
        train_df = pd.concat([train_data.drop(onehot_cols, axis=1).reset_index(drop=True), 
                              ohe_df_train.reset_index(drop=True)], axis=1)

        test_df = pd.concat([test_data.drop(onehot_cols, axis=1).reset_index(drop=True), 
                             ohe_df_test.reset_index(drop=True)], axis=1)
        
        logger.debug('Train and test dataset encoded successfully')

        # ---- Process Amenities ----
        for df in [train_df, test_df]:
            df['Amenities'] = df['Amenities'].str.replace(" ", "")
            df['Amenities_list'] = df['Amenities'].str.split(',')

        # Get unique amenities from BOTH datasets
        unique_amenities = sorted(set(
            item for df in [train_df, test_df] 
            for sublist in df['Amenities_list'] for item in sublist
        ))

        for df in [train_df, test_df]:
            for amenity in unique_amenities:
                df[f"Amenity_{amenity}"] = df['Amenities_list'].apply(lambda x: 1 if amenity in x else 0)

            df.drop(columns=['Amenities', 'Amenities_list', 'ID'], inplace=True, errors='ignore')

        logger.debug('Amenities successfully encoded for train and test')

        return train_df, test_df

    except Exception as e:
        logger.error('Error during data encoding: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise
    
def main():
    try:
        train_data = load_data('./data/raw/train.csv')
        test_data = load_data('./data/raw/test.csv')
    
        train_df, test_df = apply_labelEncoder(train_data, test_data)
        
        save_data(train_df, os.path.join("./data", "processed", "train_encoded.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_encoded.csv"))
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()    