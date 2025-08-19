import pandas as pd
import os

# --- Configuration and Data Paths ---
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, 'data') # Assumes user_mapping.csv is in a 'data' subfolder

user_mapping_file_path = os.path.join(data_folder_path, 'bike.csv')

# --- Helper to get unique options from a column that might contain comma-separated strings ---
def get_unique_flattened_options(df_column):
    """
    Extracts unique, flattened options from a pandas Series where entries
    might be single strings or comma-separated strings.
    """
    all_items = []
    for item in df_column.dropna():
        # Ensure item is treated as a string before splitting
        if isinstance(item, str):
            all_items.extend([tag.strip() for tag in item.split(',') if tag.strip()])
        # If the column might already contain lists (e.g., if preprocessed), handle that
        elif isinstance(item, list):
            all_items.extend(item)
    return sorted(list(set(all_items)))

try:
    # Load the user mapping data
    # Use 'engine=python' and 'quotechar' for robust parsing,
    # as user_mapping.csv often has complex entries with commas inside quotes.
    user_df = pd.read_csv(user_mapping_file_path, encoding='utf-8', quotechar='"', engine='python')

    # Check if 'Target_Bike_Type_Keywords' column exists after loading
    if 'Seat Height (mm)' in user_df.columns:
        print(f"--- Unique entries in 'Target_Bike_Type_Keywords' from '{user_mapping_file_path}' ---")

        unique_keywords = get_unique_flattened_options(user_df['Seat Height (mm)'])
        
        if unique_keywords:
            for keyword in unique_keywords:
                print(f"- {keyword}")
        else:
            print("No unique keywords found in 'Target_Bike_Type_Keywords' or column is empty.")
    else:
        print(f"Error: Column 'Target_Bike_Type_Keywords' not found in '{user_mapping_file_path}'.")
        print(f"Available columns are: {user_df.columns.tolist()}")

except FileNotFoundError:
    print(f"Error: User mapping data file not found at '{user_mapping_file_path}'.")
    print("Please ensure 'user_mapping.csv' is in a 'data' subfolder within the script's directory.")
except Exception as e:
    print(f"An error occurred: {e}")