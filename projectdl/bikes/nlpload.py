import pandas as pd
import os
import io 


script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, 'data')

# Construct the full path to the user personality mappings CSV file
user_mappings_csv_path = os.path.join(data_folder_path, 'user_mapping.csv')

def load_user_mappings(file_path):
    """
    Loads the user personality mappings CSV file into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', engine='python')
        print(f"'{os.path.basename(file_path)}' loaded successfully!")

        # Assign column names based on our agreed-upon 7-column structure
        df.columns = [
            "User_Description_Input",
            "Target_Bike_Type_Keywords",
            "Target_Tag_Keywords",
            "Target_Weight_Profile_Keywords",
            "Target_Height_Profile_Keywords",
            "Target_Primary_Use_Keywords",
            "Target_Riding_Style_Keywords"
        ]
        print("Columns assigned:", df.columns.tolist())
        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure 'user_personality_mappings.csv' is in the specified 'data' folder.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error parsing the CSV file '{file_path}': {e}")
        print("This often indicates issues with inconsistent rows (e.g., extra commas or unquoted fields).")
        print("Please double-check the formatting of your 'user_personality_mappings.csv' file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return None

if __name__ == "__main__":
    print("--- Loading User Personality Mappings Data ---")
    user_mappings_df = load_user_mappings(user_mappings_csv_path)

    if user_mappings_df is not None:
        print("\nFirst 5 rows of User Mappings DataFrame:")
        print(user_mappings_df.head())
        print(f"\nDataFrame shape: {user_mappings_df.shape}")
        print("\nMissing values after initial load:")
        print(user_mappings_df.isnull().sum())

        # --- Future Processing Steps for NLP Data ---
        print("\n--- Preparing Text Features for NLP ---")

        
        user_mappings_df['Combined_User_Text'] = user_mappings_df['User_Description_Input'].fillna('')

        # 2. Basic text cleaning (e.g., removing extra spaces, lowercasing)
        user_mappings_df['Combined_User_Text'] = user_mappings_df['Combined_User_Text'].str.lower().str.strip()
        user_mappings_df['Combined_User_Text'] = user_mappings_df['Combined_User_Text'].str.replace(r'[^\w\s]', '', regex=True) # Remove punctuation

        print("Created 'Combined_User_Text' for NLP processing.")
        print(user_mappings_df[['User_Description_Input', 'Combined_User_Text']].head())

       