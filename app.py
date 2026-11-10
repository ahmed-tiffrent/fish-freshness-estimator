import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 # OpenCV for Grad-CAM image manipulation

# ==================================
# ✅ Page Configuration (MUST BE FIRST ST COMMAND)
# ==================================
st.set_page_config(page_title="Fish Freshness Classification", layout="wide") # <-- MOVED HERE

# ==================================
# ✅ Configuration & Constants
# ==================================
MODEL_PATH = "final_best_model.keras" # Make sure this file is in the same directory
IMG_SIZE = (299, 299)
CLASS_NAMES = ['Fresh', 'Highly Fresh', 'Not Fresh'] # Based on {'Fresh': 0, 'Highly Fresh': 1, 'Not Fresh': 2}
LAST_CONV_LAYER_NAME = "top_conv" # From EfficientNetB3 structure used in training

# ==================================
# ✅ Model Loading (Cached)
# ==================================
# @st.cache_resource # Cache the model loading - decorator is fine here
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        # Note: It's generally better practice to not have UI elements (like st.success)
        # inside cached functions if possible, but it works.
        # A better pattern might be to load and return, then show success outside.
        st.success(f"Model '{model_path}' loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop execution if model fails to load

model = load_keras_model(MODEL_PATH) # Model loading now happens AFTER set_page_config

# ==================================
# ✅ Image Preprocessing
# ==================================
def preprocess_image(image_pil):
    """Preprocesses a PIL Image for model prediction."""
    img = image_pil.resize(IMG_SIZE)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ==================================
# ✅ Grad-CAM Functions
# ==================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates Grad-CAM heatmap."""
    # ... (rest of the function remains the same)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_gradcam(img_pil, heatmap, alpha=0.5):
    """Superimposes Grad-CAM heatmap onto the original image."""
    # ... (rest of the function remains the same)
    img = img_pil.resize(IMG_SIZE)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_cv = np.array(img)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img_cv * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# ==================================
# ✅ Streamlit App UI
# ==================================
# st.set_page_config(...) # <-- REMOVED FROM HERE

st.title("🖼️ Fish Freshness Estimator") # Title is fine here
st.write(f"Estimated results: Fresh, Highly Fresh, Not Fresh")
st.write(f"Supported species: Chanos chanos, Johnius trachycephalus, Nibea albiflora, Rastrelliger faughni, Upeneus moluccensis, Eleutheronema tetradactylum, Oreochromis mossambicus, and Oreochromis niloticus.")

# File Uploader
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing fish freshness..."):
        img_array = preprocess_image(image_pil)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions[0]) * 100

    with col2:
        st.subheader("📊 Prediction Results")
        st.success(f"Predicted Freshness: **{predicted_class_name}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Confidence Scores per Class:")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"- {class_name}: {predictions[0][i]*100:.2f}%")

    st.divider()
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap (Model Focus)", value=False)

    if show_gradcam:
        with st.spinner("Generating Grad-CAM..."):
            try:
                heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME, predicted_class_index)
                superimposed_image = superimpose_gradcam(image_pil, heatmap)
                st.subheader("🔥 Grad-CAM Visualization")
                st.image(superimposed_image, caption="Grad-CAM Heatmap Overlay", use_column_width=True)
                st.info("Grad-CAM highlights the regions in the image that were most important for the model's prediction.")
            except Exception as e:
                st.error(f"Could not generate Grad-CAM. Error: {e}")
                st.warning(f"Ensure the layer name '{LAST_CONV_LAYER_NAME}' is correct for your model.")

else:
    st.info("Please upload an image file.")

