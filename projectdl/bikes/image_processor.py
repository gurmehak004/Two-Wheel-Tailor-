# image_processor.py
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
import streamlit as st
import os
import json

# --- MediaPipe Imports for Pose Detection ---
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# --- Paths to Places365 Caffe Model and Labels File (UPDATED TO BE DIRECTLY IN 'models' FOLDER) ---
# Assuming 'models' directory contains the files directly.
PROTOTXT_PATH = os.path.join("models", "deploy_resnet152_places365.prototxt") # <<< Adjusted path
CAFFEMODEL_PATH = os.path.join("models", "resnet152_places365.caffemodel")   # <<< Adjusted path
LABELS_PATH = os.path.join("models", "categories_places365.txt")         # <<< Adjusted path


# --- Cached Function to load the scene model ---
@st.cache_resource
def load_scene_model():
    """
    Loads the pre-trained Places365 Caffe model (ResNet152) and its labels.
    """
    try:
        if not os.path.exists(PROTOTXT_PATH):
            raise FileNotFoundError(f"Prototxt file not found at: {PROTOTXT_PATH}")
        if not os.path.exists(CAFFEMODEL_PATH):
            raise FileNotFoundError(f"Caffe model file not found at: {CAFFEMODEL_PATH}")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels TXT file not found at: {LABELS_PATH}")

        # Load Caffe model
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
        
        # Set the preferable backend and target. This can sometimes improve performance.
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load labels from TXT file
        with open(LABELS_PATH, 'r') as f:
            # Each line is '/a/airfield 0', we want just 'airfield'
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


# --- Cached Function to load MediaPipe Pose model ---
@st.cache_resource
def load_pose_model():
    """Loads MediaPipe Pose model."""
    return mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)


# --- Transparency clearing function ---
def clear_transparency(img: Image.Image) -> Image.Image:
    """
    Converts an RGBA image to RGB by compositing it onto a white background.
    If the image is already RGB, it's returned as is.
    """
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, (0, 0), img)
        return background
    else:
        return img.convert('RGB')

# --- Dominant color extraction function ---
def extract_dominant_color(image_np: np.ndarray) -> str:
    """
    Extracts the dominant color from an OpenCV image (NumPy array).
    """
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
    kmeans.fit(pixels)
    dominant_rgb = kmeans.cluster_centers_[0].astype(int)

    r, g, b = dominant_rgb
    if r > 200 and g < 100 and b < 100: return "Red"
    elif r < 100 and g > 200 and b < 100: return "Green"
    elif r < 100 and g < 100 and b > 200: return "Blue"
    elif r > 200 and g > 200 and b < 100: return "Yellow"
    elif r > 200 and g < 100 and b > 200: return "Magenta"
    elif r < 100 and g > 200 and b > 200: return "Cyan"
    elif r > 150 and g > 100 and b < 50: return "Orange"
    elif r > 100 and g > 100 and b > 100 and r < 200 and g < 200 and b < 200: return "Gray"
    elif r > 240 and g > 240 and b > 240: return "White"
    elif r < 50 and g < 50 and b < 50: return "Black"
    elif r > 150 and g < 100 and b < 150: return "Pink"
    else: return f"Custom ({r},{g},{b})"


# --- Scene extraction function ---
def extract_scene(image_np: np.ndarray) -> list[str]:
    """
    Classifies the scene of an OpenCV image using the pre-trained Places365 Caffe model (ResNet152).
    Applies softmax to the model output to get correct probabilities.
    Returns a list of formatted strings with the top 5 predicted classes and their confidences.
    """
    net, labels = load_scene_model()
    if net is None or not labels:
        return ["N/A (Scene model or labels not loaded correctly)"]

    if image_np is None or image_np.size == 0:
        return ["N/A (No valid image for scene classification)"]

    # Preprocessing for Places365 ResNet152 Caffe models
    mean_values = (104, 117, 123) # BGR mean values
    scale_factor = 1.0 # No explicit scaling to 0-1 range, mean values handle it.

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image_np, scale_factor, (224, 224), mean_values, swapRB=False, crop=False)

    net.setInput(blob)
    
    # Try to get output explicitly from the 'prob' layer, which is common for Caffe classification models.
    # If 'prob' doesn't exist as an output layer, fall back to default forward pass.
    try:
        output_names = net.getUnconnectedOutLayersNames()
        if 'prob' in output_names:
            raw_outputs = net.forward('prob')
        else:
            st.warning("Could not find 'prob' as an output layer name. Attempting default forward pass.")
            raw_outputs = net.forward() # Fallback to default output layer
    except cv2.error as e:
        st.error(f"Error during model forward pass: {e}. This might indicate a problem with the model file or the output layer identification.")
        return ["Error: Could not get model output."]


    # --- APPLY SOFTMAX ACTIVATION ---
    # The 'prob' layer in Caffe models often *already* applies softmax.
    # If raw_outputs from 'prob' are already probabilities (sum to ~1), skip this.
    # However, if not explicitly sure or if falling back to another layer,
    # applying softmax here ensures we get proper probabilities.
    # Given the low initial percentages, it's safer to always apply softmax if the sum isn't close to 1.
    if np.isclose(np.sum(raw_outputs), 1.0): # Check if they are already probabilities
        probabilities = raw_outputs.flatten()
    else:
        # For numerical stability, subtract the max value before exponentiation
        exp_outputs = np.exp(raw_outputs - np.max(raw_outputs))
        probabilities = exp_outputs / np.sum(exp_outputs)
        probabilities = probabilities.flatten() # Ensure 1-D array

    # Get the top 5 predictions from the probabilities
    class_ids = np.argsort(probabilities)[::-1][:5] # Get indices of top 5 highest probabilities
    confidences = [probabilities[i] for i in class_ids] # These are now actual probabilities

    predictions = []
    for i in range(len(class_ids)):
        label = labels[class_ids[i]]
        confidence = confidences[i] * 100 # Convert to percentage (will now be 0-100)
        predictions.append(f"{label}: {confidence:.2f}%")

    return predictions


# --- Existing Pose extraction function (NO CHANGES) ---
def extract_pose(image_np: np.ndarray):
    """
    Extracts human pose landmarks from an OpenCV image using MediaPipe.
    Args:
        image_np (np.array): The image in OpenCV format (BGR).
    Returns:
        tuple: (description_str, image_with_landmarks_np)
               description_str: Text indicating if pose was detected.
               image_with_landmarks_np: Image with landmarks drawn (BGR), or None if no pose.
    """
    pose_detector = load_pose_model()
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # MediaPipe expects RGB

    # Process the image to find pose landmarks
    results = pose_detector.process(image_rgb)

    image_copy_bgr = image_np.copy() # Make a copy to draw on, preserving original

    if results.pose_landmarks:
        # Draw the pose landmarks on the image copy
        mp_drawing.draw_landmarks(
            image_copy_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return "Detected", image_copy_bgr
    else:
        return "Not Detected", None