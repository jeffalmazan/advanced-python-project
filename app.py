import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Load models
cnn_model = load_model('cnn_model.h5')
# with open('logistic_regression_model.pkl', 'rb') as f:
#     logistic_model = pickle.load(f)
# with open('decision_tree_model.pkl', 'rb') as f:
#     decision_tree_model = pickle.load(f)

# Load class indices for CNN
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
print("Class Indices Loaded:", class_indices)

# Define class labels 
# class_labels = [
#     "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
#     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
# ]

# Streamlit UI
st.title("ASL Recognition App")
st.write("Upload an image and choose a model to predict the ASL character.")

# Dropdown to select model
model_choice = st.selectbox(
    "Select a model for prediction:",
    ["CNN (MobileNetV2)"]
)

# Upload image
uploaded_file = st.file_uploader("Choose an image to upload:", type=["jpg", "png"])

def predict_outside_image_streamlit(image, model, model_choice, class_indices=None):
    # Step 1: Preprocess the uploaded image
    img_array = np.array(image.resize((128, 128))) / 255.0  # Normalize pixel values

    # Step 2: Handle grayscale images
    if img_array.ndim == 2:  # Grayscale image has 2 dimensions (height, width)
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB by duplicating channels
    
    if model_choice == "CNN (MobileNetV2)":
        # Step 3: Preprocess image for CNN
        img_array_expanded = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Step 4: Predict
        predictions = model.predict(img_array_expanded)[0]  # Get probabilities
        predicted_index = np.argmax(predictions)  # Find the class with the highest probability

        # Step 5: Map index to label
        if class_indices is not None:
            reverse_class_indices = {v: k for k, v in class_indices.items()}
            predicted_label = reverse_class_indices[predicted_index]
        else:
            raise ValueError("Class indices must be provided for CNN model.")

        # Step 6: Calculate top 5 predictions
        top_5_predictions = sorted(
            [(reverse_class_indices[i], prob) for i, prob in enumerate(predictions)],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    
    else:
        # For non-CNN models (e.g., Logistic Regression or Decision Tree)
        img_flattened = img_array.flatten().reshape(1, -1)  # Flatten image
        predicted_index = int(model.predict(img_flattened)[0])  # Predict class

        # Use reverse_class_indices for consistent label mapping
        reverse_class_indices = {v: k for k, v in class_indices.items()}
        predicted_label = reverse_class_indices[predicted_index]

        # Non-CNN models typically don't return probabilities
        top_5_predictions = [(predicted_label, 1.0)]  # Just return the predicted label

    return predicted_label, top_5_predictions



if uploaded_file is not None:
    # Load and preprocess the image
    outside_image = Image.open(uploaded_file)
    st.image(outside_image, caption="Uploaded Image", use_column_width=True)

    # Select the model for prediction
    if model_choice == "CNN (MobileNetV2)":
        model = cnn_model
        # Use the loaded class_indices
        if class_indices is None:
            raise ValueError("Class indices are not loaded. Make sure 'class_indices.pkl' exists.")
    # else:
    #     model = logistic_model if model_choice == "Logistic Regression" else decision_tree_model
    #     class_indices = None  # Not needed for non-CNN models

    # Predict the label and top 5 probabilities
    predicted_label, top_5_predictions = predict_outside_image_streamlit(
        image=outside_image,
        model=model,
        model_choice=model_choice,
        class_indices=class_indices
    )

    # Display the results
    st.subheader("Predicted Label:")
    st.write(predicted_label)

    if model_choice == "CNN (MobileNetV2)":
        st.subheader("Top 5 Class Probabilities:")
        for label, prob in top_5_predictions:
            st.write(f"{label}: {prob:.2f}")

# Footer
st.write("This app supports CNN, Logistic Regression, and Decision Tree models for ASL recognition.")
