import os
import pandas as pd
import logging

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger = logging.getLogger('feature_addde')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'featur_added.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded succesfully: %s',file_path)
        return df
    except pd.errors.ParserWarning as e:
        logger.error('Failed to parse the CSV file:%s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loaded the data:%s',e)
        raise
    
    
def feature_adder(train_data: pd.DataFrame, test_data: pd.DataFrame):

    amenity_cols = [
        'Amenity_Clubhouse', 'Amenity_Garden', 'Amenity_Gym',
        'Amenity_Playground', 'Amenity_Pool'
    ]

    def apply_rules(df: pd.DataFrame):
        df = df.copy()
        df['Investment_Score'] = 0

        # Precomputed values (dataset dependent)
        median_price = df['Price_in_Lakhs'].median()
        median_pps   = df['Price_per_SqFt'].median()
        mode_status  = df['Availability_Status'].mode()[0]
        avg_hosp     = df['Nearby_Hospitals'].mean()

        # Apply rule scoring
        df['Investment_Score'] += (df['Price_in_Lakhs'] <= median_price).astype(int)
        df['Investment_Score'] += (df['Price_per_SqFt'] <= median_pps).astype(int)
        df['Investment_Score'] += (df['BHK'] >= 3).astype(int)
        df['Investment_Score'] += (df['Availability_Status'] == mode_status).astype(int)

        # Conditional override rule
        condition = (
            (df['Nearby_Hospitals'] > avg_hosp) &
            (df['Public_Transport_Accessibility'] == 1) &
            (df[amenity_cols].sum(axis=1) > 0)
        )
        df.loc[condition, 'Investment_Score'] = 1
        
        # Final binary label
        df['Good_Investment'] = (df['Investment_Score'] >= 2).astype(int)
    
        df = df.drop(["Investment_Score"], axis=1)

        return df

    return apply_rules(train_data), apply_rules(test_data)

def save_data(df: pd.DataFrame, file_path):
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
        train_data = load_data('./data/processed/train_encoded.csv')
        test_data = load_data('./data/processed/test_encoded.csv')
    
        train_df, test_df = feature_adder(train_data, test_data)
        
        save_data(train_df, os.path.join("./data", "final", "train_encoded.csv"))
        save_data(test_df, os.path.join("./data", "final", "test_encoded.csv"))
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()    