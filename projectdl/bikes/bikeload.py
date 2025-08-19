import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

# Define the path to your data folder
script_dir = os.path.dirname(__file__)
data_folder_path = os.path.join(script_dir, 'data')

# Ensure the data folder exists
if not os.path.exists(data_folder_path):
    print(f"Error: Data folder not found at '{data_folder_path}'")
    print("Please make sure your 'bike.csv' file is located in a 'data' folder relative to your script.")
    exit()

# Construct the full path to the CSV file
bike_csv_path = os.path.join(data_folder_path, 'bike.csv')

try:
    bike_features_df = pd.read_csv(bike_csv_path, encoding='utf-8')
    print("CSV loaded successfully!")

    # --- Further Processing Steps ---

    # 1. Handle Missing Values
    print("\n--- Handling Missing Values ---")

    # Fill missing 'Engine Displacement (CC)' with the median value
    # Median is used as it's less sensitive to extreme values than the mean.
    median_engine_displacement = bike_features_df['Engine Displacement (CC)'].median()
    bike_features_df['Engine Displacement (CC)'].fillna(median_engine_displacement, inplace=True)
    print(f"Filled missing 'Engine Displacement (CC)' with median: {median_engine_displacement:.2f}")

    # Handle 'Seat Height (mm)' column:
    # First, replace 'N/A' strings with actual NaN (Not a Number) values.
    bike_features_df['Seat Height (mm)'] = bike_features_df['Seat Height (mm)'].replace('N/A', np.nan)
    # Then, attempt to convert the column to a numeric type.
    # 'errors='coerce'' will turn any values that cannot be converted (like "793 or 804") into NaN.
    bike_features_df['Seat Height (mm)'] = pd.to_numeric(bike_features_df['Seat Height (mm)'], errors='coerce')
    # Finally, fill any remaining NaN values in 'Seat Height (mm)' with its median.
    median_seat_height = bike_features_df['Seat Height (mm)'].median()
    bike_features_df['Seat Height (mm)'].fillna(median_seat_height, inplace=True)
    print(f"Filled missing 'Seat Height (mm)' with median: {median_seat_height:.2f}")

    print("\nMissing values after handling:")
    print(bike_features_df.isnull().sum())


    # 2. Prepare Text Features for Recommendation
    print("\n--- Preparing Text Features ---")

    # Combine 'Target Rider Profile' and 'Associated Tag Keywords' into a single string.
    # We use .fillna('') to ensure any potential NaN values in these columns are treated as empty strings
    # before concatenation, preventing errors.
    bike_features_df['Combined Text Features'] = bike_features_df['Target Rider Profile'].fillna('') + " " + \
                                                  bike_features_df['Associated Tag Keywords'].fillna('')

    print("Combined 'Target Rider Profile' and 'Associated Tag Keywords' into 'Combined Text Features'.")


    # 3. Apply TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization
    print("\n--- Applying TF-IDF Vectorization ---")

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=3)

    # Fit the vectorizer to your combined text features and transform them into a TF-IDF matrix.
    tfidf_matrix = tfidf_vectorizer.fit_transform(bike_features_df['Combined Text Features'])

    print(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")
    # Reverting to get_featurenames_out() as per your environment's requirement
    print(f"Number of unique features (words) extracted: {len(tfidf_vectorizer.get_feature_names_out())}")


    # 4. Calculate Cosine Similarity
    print("\n--- Calculating Cosine Similarity ---")

    # Cosine similarity measures the cosine of the angle between two non-zero vectors.
    # It's a commonly used metric to determine how similar two items are based on their features.
    # A score closer to 1 means higher similarity, closer to 0 means lower similarity.
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print(f"Cosine similarity matrix created with shape: {cosine_sim.shape}")
    print("This matrix (cosine_sim) can now be used to find similar bikes for recommendation.")


except FileNotFoundError:
    print(f"Error: The file '{bike_csv_path}' was not found.")
    print("Please ensure 'bike.csv' is in the specified 'data' folder.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{bike_csv_path}' is empty.")
except pd.errors.ParserError as e:
    print(f"Error parsing the CSV file: {e}")
    print("This often indicates issues with inconsistent rows (e.g., extra commas or unquoted fields).")
    print("Even though we checked, please double-check the formatting of your 'bike.csv' file again if this error persists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")