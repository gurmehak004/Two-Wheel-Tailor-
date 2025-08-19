import cv2
import numpy as np
import os

def classify_scene_image(image_path):
    # Define paths to the model files
    model_dir = "models/scene_classification"
    prototxt_path = os.path.join(model_dir, "deploy_resnet50_places365.prototxt")
    caffemodel_path = os.path.join(model_dir, "resnet50_places365.caffemodel")
    labels_path = os.path.join(model_dir, "categories_places365.txt")

    # Check if model files exist
    if not os.path.exists(prototxt_path):
        print(f"Error: Prototxt file not found at {prototxt_path}")
        return
    if not os.path.exists(caffemodel_path):
        print(f"Error: Caffemodel file not found at {caffemodel_path}")
        return
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        return

    # Load the pre-trained Caffe model
    print("Loading pre-trained Places365 model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    print("Model loaded successfully.")

    # Load the class labels
    with open(labels_path) as f:
        classes = [line.strip().split(' ')[0] for line in f.readlines()]
    print(f"Loaded {len(classes)} labels.")

    # Load the input image
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the path and file integrity.")
        return

    # Prepare the image for the network: resize and normalize
    # The model expects 224x224 input, mean subtraction, and channel swapping
    # (104, 117, 123) are common mean values for BGR images
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123), swapRB=False, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Perform a forward pass through the network to get the predictions
    print("Performing scene classification...")
    preds = net.forward()

    # Get the top 5 predictions
    # Sort in descending order of probabilities
    top_n = 5
    top_indices = np.argsort(preds[0])[::-1][:top_n]

    print("\n--- Top Scene Predictions ---")
    for i, idx in enumerate(top_indices):
        label = classes[idx]
        confidence = preds[0][idx] * 100  # Convert to percentage
        print(f"{i+1}. {label}: {confidence:.2f}%")

    # Optional: Display the image with the top prediction (requires GUI environment)
    # If you get errors about 'no display name' or similar, you might need to run this on a system with a GUI or skip this part.
    try:
        # Get the top prediction label
        top_label = classes[top_indices[0]]
        # Put text on the image
        output_image = image.copy()
        text = f"Scene: {top_label}"
        cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Scene Classification", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\nCould not display image (GUI might not be available or an error occurred): {e}")
        print("Classification results are printed in the console.")


if __name__ == "__main__":
    # --- BEFORE RUNNING ---
    # 1. Make sure you have an image file (e.g., test_image.jpg)
    #    in your 'your_project_folder/' or provide its full path.
    #    For example, download a sample image of a 'bedroom' or 'street'.
    # 2. Replace 'path/to/your/image.jpg' with the actual path to your test image.

    # Example usage:
    # If your image is directly in your_project_folder/
    # test_image_path = "test_image.jpg"

    # If your image is in a subfolder like 'images/' within your_project_folder/
    # test_image_path = "images/my_test_image.jpg"

    # IMPORTANT: REPLACE THIS WITH YOUR ACTUAL IMAGE PATH
    test_image_path = "path/to/your/image.jpg" # <--- REPLACE THIS LINE!

    classify_scene_image(test_image_path)