import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import cv2
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from gtts import gTTS
from sklearn.cluster import KMeans
import urllib.request

# --- MediaPipe Imports for Pose Detection ---
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Cached Function to load MediaPipe Pose model ---
@st.cache_resource
def load_pose_model():
    """Loads MediaPipe Pose model."""
    return mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

# --- Paths to Places365 Caffe Model and Labels File ---
PROTOTXT_PATH = os.path.join("models", "deploy_resnet152_places365.prototxt")
CAFFEMODEL_PATH = os.path.join("models", "resnet152_places365.caffemodel")
LABELS_PATH = os.path.join("models", "categories_places365.txt")

# Global variables for model and labels to avoid reloading
_NET = None
_LABELS = []

# --- Cached Function to load the scene model ---
@st.cache_resource
def load_scene_model():
    """Loads the pre-trained Places365 Caffe model (ResNet152) and its labels."""
    try:
        if not os.path.exists(PROTOTXT_PATH):
            raise FileNotFoundError(f"Prototxt file not found at: {PROTOTXT_PATH}")
        if not os.path.exists(CAFFEMODEL_PATH):
            raise FileNotFoundError(f"Caffe model file not found at: {CAFFEMODEL_PATH}")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels TXT file not found at: {LABELS_PATH}")

        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip().split(' ')[0].split('/')[-1] for line in f]
        
        print(f"Places365 ResNet152 Caffe model and labels loaded successfully.")
        return net, labels
    except FileNotFoundError as e:
        st.error(f"Error loading scene classification model files: {e}")
        st.info(f"Please ensure your 'models/' directory contains ALL the Places365 Caffe model files and labels directly.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading scene model: {e}")
        st.info("Check your model files' integrity and ensure OpenCV DNN module supports this Caffe model. Also, verify `categories_places365.txt` format.")
        return None, None

def clear_transparency(pil_image):
    """Converts image with alpha channel to RGB with a white background."""
    if pil_image.mode in ('RGBA', 'LA') or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
        alpha = pil_image.split()[-1]
        bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        bg.paste(pil_image, mask=alpha)
        return bg
    return pil_image

def extract_dominant_color(image_np):
    """Extracts the dominant color from an image."""
    try:
        pixels = np.float32(image_np.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 1
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = centers[0].astype(int)
        return tuple(dominant_color.tolist())
    except Exception as e:
        st.error(f"Error extracting dominant color: {e}")
        return (255, 255, 255)

def download_and_load_scene_files():
    """
    Downloads and prepares the Places365 model files and labels.
    """
    global _NET, _LABELS

    # Define file paths and URLs
    MODEL_PROTOTXT = os.path.join("models", "deploy_resnet152_places365.prototxt")
    MODEL_CAFFEMODEL = os.path.join("models", "resnet152_places365.caffemodel")
    LABELS_FILE = os.path.join("models", "categories_places365.txt")

    prototxt_url = "http://places2.csail.mit.edu/models_places365/deploy_resnet152_places365.prototxt"
    caffemodel_url = "http://places2.csail.mit.edu/models_places365/resnet152_places365.caffemodel"
    labels_url = "https://raw.githubusercontent.com/csailvision/places365/master/labels/categories_places365.txt"
    
    # Create the 'models' directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Check if files exist, download if not
    if not os.path.exists(MODEL_PROTOTXT):
        st.info(f"Downloading {MODEL_PROTOTXT}...")
        try:
            urllib.request.urlretrieve(prototxt_url, MODEL_PROTOTXT)
        except Exception as e:
            st.error(f"Failed to download model prototxt file. Please check your internet connection. Error: {e}")
            return False
    if not os.path.exists(MODEL_CAFFEMODEL):
        st.info(f"Downloading {MODEL_CAFFEMODEL}...")
        try:
            urllib.request.urlretrieve(caffemodel_url, MODEL_CAFFEMODEL)
        except Exception as e:
            st.error(f"Failed to download model caffemodel file. Please check your internet connection. Error: {e}")
            return False
    if not os.path.exists(LABELS_FILE):
        st.info(f"Downloading {LABELS_FILE}...")
        try:
            urllib.request.urlretrieve(labels_url, LABELS_FILE)
        except Exception as e:
            st.error(f"Failed to download labels file. Please check your internet connection. Error: {e}")
            return False
    
    # Load the model and labels
    try:
        _NET = cv2.dnn.readNetFromCaffe(MODEL_PROTOTXT, MODEL_CAFFEMODEL)
        with open(LABELS_FILE) as f:
            _LABELS = [line.strip() for line in f.readlines()]
        print("Model and labels loaded successfully.")
        return True
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        _NET = None
        _LABELS = []
        return False

@st.cache_resource(show_spinner="Loading scene classification model...")
def get_scene_model():
    """Wrapper to cache the model loading process."""
    if download_and_load_scene_files():
        return _NET, _LABELS
    return None, None

def extract_scene(image_np: np.ndarray) -> list[str]:
    """
    Classifies the scene of an OpenCV image using the pre-trained Places365 Caffe model (ResNet152).
    Applies softmax to the model output to get correct probabilities.
    Returns a list of formatted strings with the top 5 predicted classes and their confidences.
    """
    net, labels = get_scene_model()
    if net is None or not labels:
        return ["N/A (Scene model not loaded correctly)"]

    if image_np is None or image_np.size == 0:
        return ["N/A (No valid image for scene classification)"]

    # Preprocessing for Places365 ResNet152 Caffe models
    mean_values = (104, 117, 123) # BGR mean values
    scale_factor = 1.0 

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image_np, scale_factor, (224, 224), mean_values, swapRB=False, crop=False)

    net.setInput(blob)
    
    # Run a forward pass to get the predictions
    try:
        raw_outputs = net.forward()
        raw_outputs = raw_outputs[0]
        
    except cv2.error as e:
        st.error(f"Error during model forward pass: {e}.")
        return ["Error: Could not get model output."]

    # Apply softmax to get correct probabilities
    probabilities = np.exp(raw_outputs) / np.sum(np.exp(raw_outputs))

    # Get the top 5 predicted classes
    top_5_indices = np.argsort(probabilities)[::-1][:5]
    
    # Format the results
    extracted_scenes = []
    for i in top_5_indices:
        label = labels[i].split('/')[2] # Clean up the label string
        confidence = probabilities[i] * 100
        extracted_scenes.append(f"{label}: {confidence:.2f}%")
        
    return extracted_scenes

def extract_pose(image_np):
    """
    Extracts human pose landmarks from an OpenCV image using MediaPipe.
    Args:
        image_np (np.array): The image in OpenCV format (BGR).
    Returns:
        tuple: (description_str, image_with_landmarks_np)
               description_str: Text indicating if pose was detected.
               image_with_landmarks_np: Image with landmarks drawn (BGR), or None if no pose.
    """
    try:
        pose_detector = load_pose_model()
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            # Draw landmarks and connections on a new copy of the RGB image
            image_copy_rgb = image_rgb.copy()
            
            # Define custom drawing styles for landmarks and connections
            landmark_spec = mp_drawing.DrawingSpec(color=(255, 117, 66), thickness=2, circle_radius=2)
            connection_spec = mp_drawing.DrawingSpec(color=(255, 66, 230), thickness=2)
            
            mp_drawing.draw_landmarks(
                image_copy_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
            # Convert back to BGR for display if needed elsewhere, though Streamlit handles RGB fine
            image_with_landmarks_np = cv2.cvtColor(image_copy_rgb, cv2.COLOR_RGB2BGR)
            return "Detected", image_with_landmarks_np
        else:
            return "Not Detected", None
    except Exception as e:
        st.warning(f"Error in pose detection. This feature requires MediaPipe and OpenCV to be installed. Please run this app locally after installing them. Error: {e}")
        return "Not Detected (Module not installed)", None


# --- Configuration and Data Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(script_dir, 'data')
processed_bike_csv_path = os.path.join(data_folder_path, 'bike_processed.csv')

st.set_page_config(page_title="Two Wheel Tailor", layout="centered")

# --- Custom CSS Injection for a cleaner, more stylish UI ---
st.markdown(
    """
    <style>
    /* General body styling and font */
    body {
        font-family: 'Inter', sans-serif;
    }

    /* Container for the main content */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Input labels, now explicitly targeted for visibility */
    label, .stMarkdown, p, h1, h2, h3, .st-bb, .st-cb, .st-b9, .st-b5 {
        color: #2c3e50 !important;
    }

    /* Header and form titles */
    h1, h2, h3 {
        color: #C2185B !important;
        font-weight: 600;
    }
    h1 {
        text-align: center;
        font-size: 2.5em;
    }
    
    /* All other text elements (paragraphs, labels etc.) */
    p, label, .stMarkdown {
        color: #2c3e50 !important;
    }

    /* Target the main drag-and-drop area of the file uploader */
    [data-testid="stFileUploader"] {
        background-color: #fce4ec;
        border: 2px dashed #ffb6c1;
        border-radius: 10px;
        padding: 20px;
    }

    /* Target the text inside the drag-and-drop area */
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: #c2185b !important;
    }

    /* Target the 'Browse files' button */
    [data-testid="stFileUploader"] button {
        background-color: #ffb6c1 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px;
        padding: 8px 15px;
        font-weight: bold;
    }

    /* Adjust the 'Browse files' button on hover */
    [data-testid="stFileUploader"] button:hover {
        background-color: #ff8ba7 !important;
    }

    /* Style for the color display box */
    .color-box {
        width: 50px;
        height: 50px;
        border-radius: 5px;
        border: 1px solid #ccc;
        display: inline-block;
        vertical-align: middle;
        margin-right: 10px;
    }

    /* Style for the submit button */
    .stForm > div:last-child button {
        background-color: #C2185B;
        color: white !important;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stForm > div:last-child button:hover {
        background-color: #E91E63;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* FIX: Ensure all input text is visible */
    input[type="text"], input[type="number"] {
        color: #2c3e50 !important;
        -webkit-text-fill-color: #2c3e50 !important;
        background-color: #fff !important; /* Ensure background is light for contrast */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Functions for Scoring and Recommendation Logic ---
def get_physical_fit_score(user_height_cm, user_weight_kg, bike_seat_height_mm, bike_weight_kg):
    """
    Calculates a tiered physical fit score (0.0 to 1.0) for a bike based on user's stats.
    
    This function has been updated to be more robust, especially for very light users,
    to prevent recommendations of dangerously heavy bikes.
    """
    
    # --- NEW: Hard-stop rules for safety ---
    # If user is under 5 feet (152.4 cm) and the bike is heavy, it gets a 0 score.
    if user_height_cm < 152.4 and bike_weight_kg > 100:
        return 0.0

    # If user weighs less than 30 kg and the bike is heavy, it gets a 0 score.
    if user_weight_kg < 30 and bike_weight_kg > 80:
        return 0.0

    if pd.isna(bike_seat_height_mm) or pd.isna(bike_weight_kg):
        return 0.0
    
    user_height_mm = user_height_cm * 10
    height_ratio = bike_seat_height_mm / user_height_mm
    height_score = 0.0
    if 0.7 <= height_ratio <= 1.1:
        if 0.8 <= height_ratio <= 0.95:
            height_score = 1.0
        else:
            height_score = 0.7
    else:
        height_score = 0.0

    weight_score = 0.0
    
    if user_weight_kg <= 40 and bike_weight_kg > 100:
        weight_score = 0.0
    elif bike_weight_kg / user_weight_kg > 3.0:
        weight_score = 0.0
    elif bike_weight_kg / user_weight_kg > 2.5:
        weight_score = 0.2
    elif bike_weight_kg / user_weight_kg > 1.8:
        weight_score = 0.6
    else:
        weight_score = 1.0

    combined_score = height_score * weight_score
    
    return combined_score

def get_fused_recommendations(fused_text_prompt, bikes_df, bike_embeddings, user_height_cm, user_weight_kg, top_n=1):
    """
    Generates recommendations based on a combined physical and fused semantic score.
    
    This function has been updated to always provide a recommendation, even if the physical
    fit score for all bikes is zero.
    """
    if bikes_df.empty or bike_embeddings.size == 0:
        return []

    model = load_sentence_transformer_model()
    fused_user_embedding = model.encode([fused_text_prompt])
    similarity_scores = cosine_similarity(fused_user_embedding, bike_embeddings)[0]

    physical_scores = np.array([
        get_physical_fit_score(user_height_cm, user_weight_kg, row['Seat Height (mm)'], row['Weight (kg)'])
        for _, row in bikes_df.iterrows()
    ])
    
    # Check for bikes with a physical score > 0
    physically_fitting_indices = np.where(physical_scores > 0)[0]

    if len(physically_fitting_indices) > 0:
        # If there are any physically suitable bikes, filter the scores to only include them
        filtered_physical_scores = physical_scores[physically_fitting_indices]
        filtered_similarity_scores = similarity_scores[physically_fitting_indices]
        
        combined_scores = (0.6 * filtered_physical_scores) + (0.4 * filtered_similarity_scores)
        best_recommendation_index = physically_fitting_indices[np.argmax(combined_scores)]
    else:
        # If no bikes are a physical fit, fall back to semantic score and warn the user
        st.warning("No bikes in our database are a good physical fit for your stats. Recommending the best stylistic match.")
        best_recommendation_index = np.argmax(similarity_scores)
    
    sorted_indices = [best_recommendation_index]

    recommendations = []
    for i in sorted_indices:
        bike_data = bikes_df.iloc[i]
        recommendations.append({
            "model": bike_data['Brand & Model'],
            "image_url": bike_data['Image URL'],
            "kerb_weight": bike_data['Weight (kg)'],
            "physical_fit_score": physical_scores[i],
            "semantic_score": similarity_scores[i],
            "combined_score": (0.6 * physical_scores[i]) + (0.4 * similarity_scores[i])
        })
    return recommendations


def get_user_special_tag(q1, q2, q3):
    """
    Assigns a special tag to the user based on their personality answers.
    """
    if 'mountain person' in q3:
        return "Adventure Rider"
    elif 'beach person' in q3:
        return "Casual Cruiser"
    elif 'wander on streets' in q3:
        return "Urban Explorer"
    else:
        return "Versatile Rider"

# --- Cached Functions for Loading Data and Model ---
@st.cache_data
def load_processed_bike_data(file_path):
    """Loads the preprocessed bike data from bike_processed.csv."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        required_columns = ['Brand & Model', 'Seat Height (mm)', 'Weight (kg)', 'Image URL']
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        if not embedding_cols:
            st.error("Error: No embedding columns ('embedding_0', etc.) found in the processed data. "
                     "Please ensure 'unified.py' has been run successfully.")
            st.stop()
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Error: Required column '{col}' not found in '{os.path.basename(file_path)}'.")
                st.stop()
        
        df['Seat Height (mm)'] = pd.to_numeric(df['Seat Height (mm)'], errors='coerce')
        df['Weight (kg)'] = pd.to_numeric(df['Weight (kg)'], errors='coerce')
        
        return df, embedding_cols
    except FileNotFoundError:
        st.error(f"Error: Processed bike data file not found at '{file_path}'. "
                 "Please ensure 'unified.py' has been run to create this file.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading/processing bike data: {e}")
        st.stop()

@st.cache_resource
def load_sentence_transformer_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}. Check internet connection or model cache.")
        st.stop()

# --- Load data and model into the app ---
try:
    bike_df, embedding_cols = load_processed_bike_data(processed_bike_csv_path)
    model = load_sentence_transformer_model()
    bike_embeddings = bike_df[embedding_cols].values
except (FileNotFoundError, ValueError, RuntimeError) as e:
    st.error(f"Failed to load application data or model: {e}")
    st.info("Please ensure 'bike_processed.csv' is in a 'data' subfolder and the Sentence Transformer model is available.")
    st.stop()

# --- Streamlit UI ---
st.markdown("<h1 style='color: #C2185B; text-align: center;'>Two Wheel Tailor 🏍️</h1>", unsafe_allow_html=True)
st.markdown("Your personalized two-wheeler recommendation engine.")

with st.form("user_details_form"):
    st.markdown("<h3 style='color: #C2185B;'>Step 1: Your Profile</h3>", unsafe_allow_html=True)
    st.markdown("---") 
    
    user_name = st.text_input("Name:", placeholder="e.g., Alex Johnson")
    user_gender = st.radio(
        "Gender:",
        ('Male', 'Female', 'Non-binary', 'Prefer not to say')
    )
    col1, col2 = st.columns(2)
    with col1:
        feet = st.number_input("Height (Feet)", min_value=1, max_value=8, value=5)
    with col2:
        inches = st.number_input("Height (Inches)", min_value=0, max_value=11, value=7)
    user_weight_kg = st.number_input("Weight (in kg)", min_value=20, max_value=300, value=70)

    st.markdown("<h3 style='color: #C2185B;'>Step 2: Your Vibe</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("Answer a few fun questions to help us understand your style.")

    q1 = st.radio(
        "1) You would like to be a:",
        ('a) Solo Rider', 'b) Better with Family', 'c) Always Friends')
    )
    q2 = st.radio(
        "2) Honest opinion:",
        ('a) I am an introvert', 'b) I am a moodyvert', 'c) I am extremely extrovert')
    )
    q3 = st.radio(
        "3) What's your type?",
        ('a) I am a mountain person', 'b) I am a beach person', 'c) I like to wander on streets etc')
    )
    
    st.markdown("<h3 style='color: #C2185B;'>Step 3: Your Style</h3>", unsafe_allow_html=True)
    st.markdown("---")
    # UPDATED: Changed instruction from "bike style" to "yourself"
    st.write("Upload a picture of yourself so we can analyze your style and pose.")
    
    uploaded_file = st.file_uploader(
        "Upload your image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    submit_button = st.form_submit_button("Find My Perfect Match! 🚀")

# --- Recommendation Logic and Output (Now combined and conditional on submit) ---
if submit_button:
    if not all([user_name, feet is not None, inches is not None, user_weight_kg is not None, q1, q2, q3, uploaded_file]):
        st.warning("Please fill in all the details, including uploading an image, to get your recommendation.")
    else:
        dominant_color_rgb = (255, 255, 255)
        dominant_color_hex = '#FFFFFF'
        extracted_scenes = []
        pose_description = "No specific human pose was detected. (Feature in development)"
        pose_image = None
        recommendations = []
        user_tag = "N/A"

        with st.spinner("Analyzing your vibe and crunching numbers..."):
            try:
                pil_image = Image.open(uploaded_file)
                st.subheader("Original Image:")
                st.image(pil_image, caption="Original Image", use_container_width=True)
                
                processed_image_pil = clear_transparency(pil_image)
                image_np_rgb = np.array(processed_image_pil)

                dominant_color_rgb_temp = extract_dominant_color(cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))
                if isinstance(dominant_color_rgb_temp, tuple) and len(dominant_color_rgb_temp) == 3:
                    dominant_color_rgb = dominant_color_rgb_temp
                    dominant_color_hex = '#%02x%02x%02x' % dominant_color_rgb
                else:
                    st.warning("Error extracting dominant color. Using default white.")

                # Use the new extract_scene function
                extracted_scenes = extract_scene(cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR))
                
                pose_description, pose_image = extract_pose(cv2.cvtColor(image_np_rgb, cv2.COLOR_BGR2RGB))
                
                total_inches = (feet * 12) + inches
                height_cm = total_inches * 2.54

                user_tag = get_user_special_tag(q1, q2, q3)

                # Use the top scene prediction for the prompt
                top_scene = extracted_scenes[0].split(':')[0].strip() if extracted_scenes else "unspecified"

                user_answers_text = (
                    f"The user is a {q1.replace('a) ', '').replace('b) ', '').replace('c) ', '')}. "
                    f"They feel they are {q2.replace('a) ', '').replace('b) ', '').replace('c) ', '')}. "
                    f"They like to wander on {q3.replace('a) I am a ', '').replace('b) I am a ', '').replace('c) I like to wander on ', '')}. "
                    f"Their image shows a dominant color of RGB({dominant_color_rgb}), a scene of '{top_scene}', and the person's pose indicates a style of '{pose_description}'."
                )
                
                recommendations = get_fused_recommendations(
                    user_answers_text,
                    bike_df,
                    bike_embeddings,
                    height_cm,
                    user_weight_kg,
                    top_n=1
                )
            except Exception as e:
                st.error(f"An error occurred during the recommendation process: {e}")
                st.exception(e)
                
        st.success(f"Hello, {user_name}!")
        
        st.header("📊 Image Analysis Report")
        st.markdown("---")
        st.markdown(f"**Transparency Removal:** The uploaded image's transparency has been successfully handled and it's ready for analysis.")
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <div class="color-box" style="background-color: {dominant_color_hex};"></div>
                **Dominant Color:** <code>{dominant_color_hex}</code> (RGB: {dominant_color_rgb})
            </div>
            """,
            unsafe_allow_html=True
        )
        # Display the full list of predicted scenes
        st.markdown(f"**Top 5 Scene Predictions:**")
        if extracted_scenes:
            for scene in extracted_scenes:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- {scene}")
        else:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;- N/A (Could not classify scene)")
        
        st.markdown(f"**Pose Estimation:** `{pose_description}`")
        if pose_image is not None:
            st.image(cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB), caption="Pose Landmarks Detected", use_container_width=True)

        st.markdown("---")

        st.markdown("<h3 style='color: #C2185B;'>🎉 Your Perfect Match</h3>", unsafe_allow_html=True)
        
        if recommendations:
            rec = recommendations[0]
            
            tag_text = f"Your personality tag is: {user_tag}"
            bike_text = f"We recommend the {rec['model']}"
            full_text = f"{tag_text}. {bike_text}"
            
            tts = gTTS(text=full_text, lang='en')
            audio_bytes_io = io.BytesIO()
            tts.write_to_fp(audio_bytes_io)
            st.audio(audio_bytes_io, format='audio/mp3', autoplay=True)

            st.markdown(f"<p style='font-weight: bold; font-size: 1.2em; color: #C2185B;'>Your Personality Tag: {user_tag}</p>", unsafe_allow_html=True)
            st.markdown(f"#### {rec['model']}")
            st.write(f"**Kerb Weight:** {rec['kerb_weight']:.1f} kg")
            st.write(f"**Overall Score:** `{rec['combined_score']:.2f}` (Physical: {rec['physical_fit_score']:.2f} | Semantic: {rec['semantic_score']:.2f})")
            
            image_url = rec.get('image_url')
            if image_url and isinstance(image_url, str):
                try:
                    st.image(image_url, caption=rec['model'], width=300)
                except Exception as e:
                    st.error(f"Failed to load image from URL: {image_url}")
                    st.warning("Displaying a placeholder image instead.")
                    st.image("https://placehold.co/300x200/505050/FFFFFF?text=Image+Not+Found", caption=f"{rec['model']} (Image Placeholder)", width=300)
            else:
                st.image("https://placehold.co/300x200/505050/FFFFFF?text=Image+Not+Found", caption=f"{rec['model']} (Image Placeholder)", width=300)
        else:
            st.info("No recommendations could be generated with your provided information.")
        
        st.markdown("---")
    
st.markdown("---")
st.caption("Developed for bike recommendation demonstration.")
