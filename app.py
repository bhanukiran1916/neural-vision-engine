import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Professional Page Config
st.set_page_config(
    page_title="Neural Vision Engine | AI Detection",
    page_icon="🤖",
    layout="wide"
)

# 2. Modern UI CSS (Branding teesi professional ga marchanu)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .main-header {
        background: linear-gradient(90deg, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-text {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid #334155;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Header Section (No personal names)
st.markdown('<h1 class="main-header">NEURAL VISION ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Next-Generation Real-time Object Identification System</p>', unsafe_allow_html=True)

# 4. Model Loading
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# 5. Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/artificial-intelligence.png")
    st.title("Settings")
    st.markdown("---")
    confidence = st.slider("Model Sensitivity", 0.0, 1.0, 0.45)
    st.info("Status: AI Engine Ready")

# 6. Main Interface Logic
uploaded_file = st.file_uploader("Upload an image to analyze", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📥 Input Source")
        st.image(image, use_container_width=True)
        
    with st.spinner('⚙️ Processing Neural Layers...'):
        results = model(img_array, conf=confidence)
        res_plotted = results[0].plot()
        
    with col2:
        st.markdown("### 🎯 Detection Output")
        st.image(res_plotted, use_container_width=True)

    # Detections Summary
    st.markdown("---")
    st.markdown("### 📊 Detection Analysis")
    
    if len(results[0].boxes) > 0:
        cols = st.columns(3)
        for i, box in enumerate(results[0].boxes):
            with cols[i % 3]:
                label = model.names[int(box.cls[0])]
                conf_score = float(box.conf[0])
                st.markdown(f"""
                <div class="card">
                    <h4 style="margin:0; color:#60a5fa;">{label.capitalize()}</h4>
                    <p style="margin:0; color:#94a3b8;">Accuracy: {conf_score:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.write("") # Spacing
    else:
        st.warning("No objects identified with current sensitivity.")
else:
    st.markdown("""
    <div style="text-align:center; padding:50px; border:2px dashed #334155; border-radius:20px;">
        <h3 style="color:#4b5563;">Waiting for Input Assets...</h3>
        <p style="color:#4b5563;">Upload a photo in the section above to begin processing.</p>
    </div>
    """, unsafe_allow_html=True)