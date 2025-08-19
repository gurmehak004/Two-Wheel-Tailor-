# Two-Wheel-Tailor-
Two Wheel Tailor is an innovative project designed to provide personalized recommendations for two-wheelers (motorcycles and scooters) using a multi-modal approach. 
Technologies Used
Python: The core programming language for the entire project.

Streamlit: For building the interactive web application interface.

Sentence Transformers: To create high-quality embeddings for text queries.

OpenCV: For foundational image processing tasks.

MediaPipe: For advanced computer vision feature extraction.

Pandas & NumPy: For data manipulation and numerical operations.
To train and test this multi-modal system effectively, we created our own comprehensive dataset. This process involved:

Manual Data Collection: We meticulously gathered a large number of images and technical specifications for various two-wheelers from diverse sources.

Feature Annotation: Each entry in the dataset was manually annotated with specific, relevant features, including bike type (e.g., sport, cruiser, scooter), engine capacity, style attributes (e.g., vintage, futuristic), and other key characteristics. This structured annotation was crucial for the model to learn the associations between visual elements, text, and technical data.

**Important: in the bikes/models folder , add resnet152_places365 file from github under CSAILVision/places365 repositiries . This repository contains various convolutional neural networks (CNNs) trained on the Places365 dataset, which is a dataset for scene recognition. The ResNet152-places365 model, a specific model trained on this dataset, is referenced within the repository's documentation
