import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Make the webpage wide and set the title
st.set_page_config(layout="wide", page_title="Wafer Defect AI")

st.title("Semiconductor Wafer Defect Analyzer")

# 1. Load data and model (Using Streamlit cache so it doesn't reload the massive model every click)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('wafer_model.keras')

@st.cache_data
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

# 2. UI Layout (Two columns)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Select a Defect Type")
    # Radio buttons for the user to select the defect they want to test
    selected_defect = st.radio("Choose a pattern to feed to the AI:", list(defect_labels.values()))
    
    # The action button
    test_button = st.button("Analyze Wafer", type="primary")

with col2:
    if test_button:
        st.subheader("2. AI Analysis Results")
        
        # Find the ID for the selected defect
        target_id = [k for k, v in defect_labels.items() if v == selected_defect][0]
        matching_indices = np.where(y_test == target_id)[0]
        
        if len(matching_indices) > 0:
            # Pick a random wafer of that specific type
            specific_index = random.choice(matching_indices)
            test_image = X_test[specific_index]
            
            # AI makes its prediction
            prediction_prob = model.predict(np.array([test_image]))
            predicted_label_id = np.argmax(prediction_prob)
            predicted_label = defect_labels[predicted_label_id]
            confidence = np.max(prediction_prob) * 100
            
            # Create a small layout to show the image next to the text
            subcol1, subcol2 = st.columns([1, 1])
            
            with subcol1:
                # Draw the wafer map
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(test_image.reshape(64, 64))
                ax.axis('off')
                st.pyplot(fig)
                
            with subcol2:
                # Display the results
                st.write(f"**Actual Pattern:** {selected_defect}")
                
                # Color code the result (Green for correct, Red for wrong)
                if predicted_label == selected_defect:
                    st.success(f"**AI Prediction:** {predicted_label} ({confidence:.1f}%)")
                else:
                    st.error(f"**AI Prediction:** {predicted_label} ({confidence:.1f}%)")
        else:
            st.warning("No test images found for this specific defect!")