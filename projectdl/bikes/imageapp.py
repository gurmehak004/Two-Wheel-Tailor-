import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2

# Import all necessary functions from image_processor
# Ensure extract_scene is imported
from image_processor import clear_transparency, extract_dominant_color, extract_scene, extract_pose

st.set_page_config(page_title="Image Transparency Test", layout="centered")

# --- Custom CSS Injection for File Uploader (NO CHANGES) ---
st.markdown(
    """
    <style>
    /*
    IMPORTANT: Streamlit's internal CSS class names can sometimes change.
    If the styling doesn't apply perfectly, you might need to
    right-click on the 'Drag and drop file here' area in your browser
    and select 'Inspect' to find the exact class names Streamlit is using
    in your environment (look for classes like 'st-emotion-cache-...').
    */

    /* Target the main drag-and-drop area of the file uploader */
    [data-testid="stFileUploader"] {
        background-color: #FCE4EC; /* Very light pink background */
        border: 2px dashed #FFB6C1; /* Light pink dashed border */
        border-radius: 10px;
        padding: 20px; /* Add some internal spacing */
    }

    /* Target the text inside the drag-and-drop area (like "Drag and drop file here") */
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] small {
        color: #C2185B !important; /* Darker pink for the text */
        /* font-weight: bold; */ /* You can uncomment this if you want it bold */
    }

    /* Target the 'Browse files' button */
    [data-testid="stFileUploader"] button {
        background-color: #FFB6C1 !important; /* Light pink button background */
        color: white !important; /* White text on the button */
        border: none !important; /* Remove default button border */
        border-radius: 5px;
        padding: 8px 15px;
    }

    /* Adjust the 'Browse files' button on hover */
    [data-testid="stFileUploader"] button:hover {
        background-color: #FF8BA7 !important; /* Slightly darker pink on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- End Custom CSS Injection ---

st.title("Image Transparency Test App")
st.write("This app allows you to test the image upload, transparency handling, and feature extraction.")

st.markdown("---")

# --- Your Custom Instructions for Upload (NO CHANGES) ---
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 10px;">
        <h3 style="color:#FF69B4; margin-bottom: 5px;">
            💖 Drop Your Image Here! 📸
        </h3>
        <p style="color:#808080; font-size: 1.1em;">
            Simply drag & drop your photo or click 'Browse files' below.<br>
            <small>Supported: JPG, JPEG, PNG (Max 200MB per file).</small>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# The actual st.file_uploader (NO CHANGES)
uploaded_file = st.file_uploader(
    "Upload your image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        # Read the uploaded file into a PIL Image object
        image = Image.open(uploaded_file)

        st.subheader("Original Image:")
        st.image(image, caption="Original Image", use_container_width=True)
        if image.mode == 'RGBA':
            st.info("ℹ️ Original image has transparency (RGBA mode). Watch how it changes!")
        else:
            st.info(f"ℹ️ Original image is in {image.mode} mode (likely no transparency).")


        st.subheader("Processed Image (Transparency Cleared):")
        processed_image = clear_transparency(image)
        st.image(processed_image, caption="Processed Image (Transparency Handled)", use_container_width=True)
        st.success("✅ Transparency successfully processed! Look closely if your original had transparent parts.")

        st.markdown("---")
        st.subheader("Extracted Image Features:")

        # Convert PIL Image to OpenCV format (NumPy array, BGR)
        # This is where the image becomes a NumPy array (image_np) for subsequent functions
        opencv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)

        # --- Extract Dominant Color (NO CHANGES) ---
        dominant_color = extract_dominant_color(opencv_image)
        st.write(f"**Dominant Color:** {dominant_color}")

        # --- Extract Pose (NO CHANGES) ---
        pose_description, pose_image = extract_pose(opencv_image)
        st.write(f"**Pose:** {pose_description}")
        if pose_image is not None:
            # MediaPipe draws on BGR image, Streamlit expects RGB for display
            st.image(cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB), caption="Pose Landmarks Detected", use_container_width=True)

        # --- Extract Scene (UPDATED DISPLAY) ---
        # Call the extract_scene function, which now returns a list of strings
        scene_predictions = extract_scene(opencv_image) 
        st.write("**Scene Analysis (Top 5 Predictions):**")
        if scene_predictions:
            for prediction_str in scene_predictions:
                st.write(f"- {prediction_str}")
        else:
            st.write("No scene predictions available.")


    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")
        st.exception(e) # Display full traceback for easier debugging

st.markdown("---")
st.caption("Developed for image processing demonstration.")