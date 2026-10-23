
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# ==========================================
# 1. TensorFlow Grad-CAM Implementation
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates the Grad-CAM heatmap using TensorFlow."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy(), int(pred_index)

def overlay_cam_on_image(img_pil, mask):
    """Overlays the heatmap onto the original PIL image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img_np = np.array(img_pil)
    if heatmap.shape[:2] != img_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
    cam = np.float32(heatmap) / 255.0 + np.float32(img_np) / 255.0
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# ==========================================
# 2. Load Custom Keras Model & Metadata
# ==========================================
@st.cache_resource
def load_custom_model():
    """Loads the model from config.json and model.weights.h5, and classes from metadata.json."""
    
    # 1. Load Model Architecture from config.json
    with open("config.json", "r") as f:
        model_config = json.load(f)
    
    model = tf.keras.models.model_from_json(json.dumps(model_config))
    
    # 2. Load Weights from model.weights.h5
    model.load_weights("model.weights.h5")
    
    # 3. Load Class Names from metadata.json
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    classes = metadata.get("classes", metadata.get("labels", metadata.get("class_names", ["Class 0", "Class 1", "Class 2"])))
    
    # 4. Find the last Conv2D layer for Grad-CAM automatically
    # For your model, this will correctly identify 'conv5_block16_2_conv'
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        st.error("Could not find a Conv2D layer in your model for Grad-CAM.")
        st.stop()

    return model, classes, last_conv_layer_name

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="Custom DenseNet Grad-CAM", layout="wide")
st.title("🔍 Custom Model Explainability (Grad-CAM)")
st.markdown("This app uses your custom 3-class DenseNet-style model (299x299 input) to predict and visualize attention.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Loading model and processing..."):
            model, classes, last_conv_layer_name = load_custom_model()
            
            # --- PREPROCESSING ---
            # Your model expects 299x299 input based on the config.json
            IMG_SIZE = (299, 299) 
            
            img_resized = image.resize(IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img_resized)
            
            # IMPORTANT: Adjust this preprocessing to match your training pipeline!
            # Option A: If you trained with standard Keras DenseNet preprocessing, uncomment the next line:
            img_array = tf.keras.applications.densenet.preprocess_input(img_array)
            
            # Option B: Standard scaling to [0, 1] (safe default for most custom models)
            # img_array = img_array / 255.0 
            
            img_batch = np.expand_dims(img_array, axis=0)

            # 1. Get Predictions
            predictions = model.predict(img_batch, verbose=0)
            probs = tf.nn.softmax(predictions[0]).numpy()
            
            # Get all predictions sorted by probability
            top_indices = np.argsort(probs)[::-1]
            top_probs = probs[top_indices]
            top_classes = [classes[i] for i in top_indices]
            
            # 2. Generate Grad-CAM
            heatmap, pred_index = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
            
            # Create overlay image
            overlay_img = overlay_cam_on_image(image, heatmap)

        # Display Results
        with col2:
            st.subheader("Grad-CAM Heatmap Overlay")
            st.image(overlay_img, use_column_width=True)
            
        st.subheader("Predictions")
        
        pred_data = []
        for i in range(len(top_indices)):
            pred_data.append({
                "Class": top_classes[i],
                "Probability": f"{top_probs[i] * 100:.2f}%"
            })
        st.table(pred_data)
        
        st.info(f"**Model's Primary Focus:** The model classified this image as **{classes[pred_index]}**. The red/yellow areas show the pixels that drove this decision.")