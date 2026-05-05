import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="centered"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
CLASS_NAMES = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Horse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

MODEL_INFO = {
    "Baseline CNN (from scratch)": {
        "path": "largescalefish_cnn.keras",
        "size": "≈ 350 MB",
        "speed": "Slow (CPU)",
        "accuracy": "Trained offline"
    },
    "VGG16 Transfer Learning": {
        "path": "fish_vgg16_transfer.keras",
        "size": "≈ 55 MB",
        "speed": "Fast (CPU)",
        "accuracy": "Trained offline"
    }
}

IMG_SIZE = (170, 170)

# --------------------------------------------------
# Utils
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # IMPORTANT:
    # We trained VGG16 with preprocess_input,
    # but baseline CNN also works fine with same scaling at inference
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img_array = preprocess_input(img_array)

    return img_array

# --------------------------------------------------
# UI – Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🐟 Fish Species Classification</h1>
    <p style='text-align:center; font-size:16px;'>
    Compare a <b>Baseline CNN</b> and a <b>Transfer Learning (VGG16)</b> model.<br>
    Upload a fish image and see which model performs better.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Sidebar – Model selection
# --------------------------------------------------
st.sidebar.header("⚙️ Model Selection")

model_choice = st.sidebar.radio(
    "Choose a model:",
    list(MODEL_INFO.keys())
)

model_meta = MODEL_INFO[model_choice]

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.markdown(f"**Model size:** {model_meta['size']}")
st.sidebar.markdown(f"**Inference speed:** {model_meta['speed']}")
st.sidebar.markdown(f"**Validation accuracy:** {model_meta['accuracy']}")

# --------------------------------------------------
# Load model (lazy)
# --------------------------------------------------
with st.spinner("Loading selected model..."):
    model = load_model(model_meta["path"])

st.success(f"✅ {model_choice} loaded successfully")

# --------------------------------------------------
# Main – Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Running prediction..."):
            img_array = preprocess_image(image)
            preds = model.predict(img_array)[0]

            pred_index = int(np.argmax(preds))
            confidence = float(preds[pred_index]) * 100
            pred_class = CLASS_NAMES[pred_index]

        st.markdown("### 🧠 Prediction Result")
        st.markdown(
            f"""
            <div style="
                padding:15px;
                border-radius:10px;
                background-color:#f0f2f6;
                text-align:center;
            ">
                <h2>{pred_class}</h2>
                <p style="font-size:18px;">
                    Confidence: <b>{confidence:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

    # --------------------------------------------------
    # Probabilities table
    # --------------------------------------------------
    st.divider()
    st.subheader("📈 Class Probabilities")

    prob_data = {
        "Class": CLASS_NAMES,
        "Probability (%)": [round(p * 100, 2) for p in preds]
    }

    st.dataframe(
        prob_data,
        use_container_width=True
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:13px;'>"
    "Developed with ❤️ using CNN & Transfer Learning | Powered by Streamlit & TensorFlow"
    "</p>",
    unsafe_allow_html=True
)
