import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageOps

# Make the webpage wide and set the title
st.set_page_config(layout="wide", page_title="Wafer Defect AI")

st.title("Semiconductor Wafer Defect Analyzer")

# 1. Load data and model safely
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('wafer_model.keras')

@st.cache_data
def process_real_photo(image):
    # 1. Convert PIL image to OpenCV format (BGR)
    file_bytes = np.asarray(image.convert("RGB"))
    img = cv2.cvtColor(file_bytes, cv2.COLOR_RGB2BGR)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Enhance Contrast (CLAHE) - This makes the scratch pop!
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. Binary Thresholding (Turning it into 0s and 1s)
    # We use Otsu's method to automatically find the best 'cutoff' for the scratch
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 5. Resize to 64x64 to match our AI's training
    resized = cv2.resize(thresh, (64, 64), interpolation=cv2.INTER_AREA)
    
    # 6. Normalize (AI likes values between 0 and 1)
    normalized = resized / 255.0
    return normalized.reshape(1, 64, 64, 1)
def load_data():
    X = np.load('X_test_web.npy')
    y = np.load('y_test_web.npy')
    return X, y

model = load_model()
X_test, y_test = load_data()

defect_labels = {
    0: 'Center', 1: 'Donut', 2: 'Edge-Loc', 3: 'Edge-Ring', 
    4: 'Loc', 5: 'Near-full', 6: 'Random', 7: 'Scratch', 8: 'none'
}

# --- Create Two Tabs for the UI ---
tab1, tab2 = st.tabs(["📚 Test from Dataset", "⬆️ Upload Your Own Image"])

# --- TAB 1: The Original Dataset Tester ---
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1. Select a Defect Type")
        selected_defect = st.radio("Choose a pattern to feed to the AI:", list(defect_labels.values()))
        test_button = st.button("Analyze Wafer", type="primary")

    with col2:
        if test_button:
            st.subheader("2. AI Analysis Results")
            target_id = [k for k, v in defect_labels.items() if v == selected_defect][0]
            matching_indices = np.where(y_test == target_id)[0]
            
            if len(matching_indices) > 0:
                specific_index = random.choice(matching_indices)
                test_image = X_test[specific_index]
                
                prediction_prob = model.predict(np.array([test_image]))
                predicted_label_id = np.argmax(prediction_prob)
                predicted_label = defect_labels[predicted_label_id]
                confidence = np.max(prediction_prob) * 100
                
                subcol1, subcol2 = st.columns([1, 1])
                
                with subcol1:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(test_image.reshape(64, 64))
                    ax.axis('off')
                    st.pyplot(fig)
                    
                with subcol2:
                    st.write(f"**Actual Pattern:** {selected_defect}")
                    if predicted_label == selected_defect:
                        st.success(f"**AI Prediction:** {predicted_label} ({confidence:.1f}%)")
                    else:
                        st.error(f"**AI Prediction:** {predicted_label} ({confidence:.1f}%)")
            else:
                st.warning("No test images found for this specific defect!")

# --- TAB 2: The New Image Uploader ---
with tab2:
    st.subheader("Upload a Custom Wafer Map")
    st.write("Upload any image of a semiconductor wafer. The AI will shrink it, convert it to grayscale, and analyze the defect pattern.")
    
    # The drag-and-drop uploader
    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # 1. Open the user's image
        user_image = Image.open(uploaded_file)
        
        # 2. Preprocess it to match the AI's training (64x64, grayscale)
        processed_image = ImageOps.grayscale(user_image)
        processed_image = processed_image.resize((64, 64))
        
        # 3. Convert it to a math array and format it for the neural network
        img_array = np.array(processed_image)
        input_image = img_array.reshape(1, 64, 64, 1)
        
        # 4. Make the prediction
        prediction_prob = model.predict(input_image)
        predicted_label_id = np.argmax(prediction_prob)
        predicted_label = defect_labels[predicted_label_id]
        confidence = np.max(prediction_prob) * 100
        
        # 5. Show the results
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(user_image, caption="Your Uploaded Image", width=300)
        with col2:
            st.write("### AI Analysis")
            st.info(f"**Predicted Pattern:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.1f}%")

