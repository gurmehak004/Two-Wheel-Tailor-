import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# --- Configuration and Data Paths ---
# This script assumes 'bike.csv' and 'user_mappings.csv' are in a 'data' subfolder.
# It will generate 'bike_processed.csv' and 'user_profile_embedding.npy' in the same folder.
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, 'data')
bike_file_path = os.path.join(data_folder_path, 'bike.csv')
user_mappings_file_path = os.path.join(data_folder_path, 'user_mapping.csv')
output_bike_file_path = os.path.join(data_folder_path, 'bike_processed.csv')
output_user_profile_path = os.path.join(data_folder_path, 'user_profile_embedding.npy')

# It is crucial to have the sentence-transformers library installed.
# You may need to run: pip install sentence-transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_column_name(df, possible_names):
    """
    Finds the correct column name from a list of possibilities, ignoring case and spaces.
    """
    df_columns = [col.strip().lower() for col in df.columns]
    for name in possible_names:
        if name.strip().lower() in df_columns:
            return df.columns[df_columns.index(name.strip().lower())]
    return None

def process_data_with_transformers():
    """
    Reads raw bike and user data, processes them, generates embeddings using a
    SentenceTransformer, and saves the unified data for the recommendation engine.
    """
    try:
        # --- 1. Load Data ---
        print("Loading data...")
        bike_df = pd.read_csv(bike_file_path)
        user_mappings_df = pd.read_csv(user_mappings_file_path)

        print("Columns found in 'bike.csv':", bike_df.columns.tolist())
        print("Columns found in 'user_mappings.csv':", user_mappings_df.columns.tolist())

        # --- 2. Process Bike Data ---
        print("Processing bike data...")
        
        # Find the correct column names for Seat Height and Weight
        seat_height_col = find_column_name(bike_df, ['Seat Height (mm)'])
        weight_col = find_column_name(bike_df, ['Weight (Kerb weight in kg)', 'Weight (kg)'])

        if not seat_height_col:
            raise ValueError("Could not find 'Seat Height (mm)' column in bike.csv.")
        if not weight_col:
            raise ValueError("Could not find 'Weight' column in bike.csv. "
                             "Tried 'Weight (Kerb weight in kg)' and 'Weight (kg)'.")
            
        # Ensure numerical columns are correctly typed
        bike_df[seat_height_col] = pd.to_numeric(bike_df[seat_height_col], errors='coerce')
        bike_df[weight_col] = pd.to_numeric(bike_df[weight_col], errors='coerce')
        
        # Fill missing numerical values with the column's median
        bike_df[seat_height_col] = bike_df[seat_height_col].fillna(bike_df[seat_height_col].median())
        bike_df[weight_col] = bike_df[weight_col].fillna(bike_df[weight_col].median())
        
        # Create a combined text feature for embedding
        bike_df['Combined_Bike_Features_Text'] = bike_df['Brand & Model'].fillna('') + '. ' + \
                                                  bike_df['Bike Type'].fillna('') + '. ' + \
                                                  bike_df['Target Rider Profile'].fillna('') + '. ' + \
                                                  bike_df['Associated Tag Keywords'].fillna('')
        
        # Generate embeddings for each bike's text features using the SentenceTransformer model
        print("Generating embeddings for bike features...")
        bike_embeddings = model.encode(bike_df['Combined_Bike_Features_Text'].tolist())
        
        # Create a new DataFrame with the bike embeddings
        bike_embeddings_df = pd.DataFrame(bike_embeddings)
        bike_embeddings_df.columns = [f'embedding_{i}' for i in range(bike_embeddings_df.shape[1])]
        
        # Combine the original bike data with its new embeddings
        final_bike_df = pd.concat([bike_df, bike_embeddings_df], axis=1)
        
        # Rename the weight column for a standardized output
        final_bike_df = final_bike_df.rename(columns={weight_col: 'Weight (kg)'})
        
        # --- 3. Process User Mappings for a Single User Profile ---
        print("Processing user mappings to create a single user profile...")
        
        # We'll concatenate all user keywords into a single string
        user_preference_text = ''
        if not user_mappings_df.empty:
            # Assumes the first row contains the single user's preferences
            first_user_row = user_mappings_df.iloc[0]
            
            # Concatenate all relevant columns into a single descriptive string
            columns_to_combine = [
                'User_Description_Input', 'Target_Bike_Type_Keywords', 'Target_Tag_Keywords',
                'Target_Weight_Profile_Keywords', 'Target_Height_Profile_Keywords',
                'Target_Primary_Use_Keywords', 'Target_Riding_Style_Keywords'
            ]
            
            combined_user_text = ' '.join(
                str(first_user_row[col]) for col in columns_to_combine if pd.notnull(first_user_row[col])
            )
            user_preference_text = combined_user_text
        
        # Generate a single embedding for this user's profile
        print("Generating embedding for the user profile...")
        user_profile_embedding = model.encode(user_preference_text)
        
        # --- 4. Save Processed Data ---
        print("Saving processed data...")
        
        # Save the bike data (including original features and new embeddings)
        final_bike_df.to_csv(output_bike_file_path, index=False, encoding='utf-8')
        
        # Save the single user profile embedding
        np.save(output_user_profile_path, user_profile_embedding)
        
        print(f"Successfully processed data.")
        print(f"Unified bike data saved to: {output_bike_file_path}")
        print(f"User profile embedding saved to: {output_user_profile_path}")
        print(f"Total bikes processed: {len(final_bike_df)}")
        
    except FileNotFoundError:
        print("Error: One of the source files not found. Ensure 'bike.csv' and "
              "'user_mappings.csv' exist in the 'data' subfolder.")
    except ValueError as e:
        print(f"Data processing error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure a 'data' directory exists
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    process_data_with_transformers()

