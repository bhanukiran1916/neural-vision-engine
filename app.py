import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter
import cv2
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Neural Vision Engine | AI Monitoring",
    page_icon="🤖",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e1e2f, #121212);
    color: white;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Neural Vision Engine")
st.subheader("AI-powered real-time intelligent monitoring system")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    confidence = st.slider("Confidence", 0.1, 1.0, 0.4)

    mode = st.selectbox("Select Mode", [
        "General Detection",
        "Industrial Safety",
        "Traffic Monitoring"
    ])

    input_mode = st.radio("Input Mode", ["Image Upload", "Webcam"])

# ---------------- SESSION DATA ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "data" not in st.session_state:
    st.session_state.data = []

# =========================================================
# ================= IMAGE UPLOAD MODE ======================
# =========================================================
if input_mode == "Image Upload":

    uploaded_files = st.file_uploader(
        "Upload Image(s)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:

            image = Image.open(uploaded_file)
            img_array = np.array(image)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Input Image")

            # -------- DETECTION --------
            results = model(img_array, conf=confidence)
            res_plotted = results[0].plot()

            with col2:
                st.image(res_plotted, caption="Detection Output")

            # -------- CLASSES --------
            detected_classes = [
                model.names[int(box.cls[0])]
                for box in results[0].boxes
            ]

            counts = Counter(detected_classes)

            # -------- ANALYTICS --------
            st.markdown("## 📊 Analytics")

            colA, colB, colC = st.columns(3)
            colA.metric("Total Objects", len(detected_classes))
            colB.metric("Unique Objects", len(counts))
            colC.metric("Most Common", max(counts, key=counts.get) if counts else "None")

            for obj, count in counts.items():
                st.write(f"🔹 {obj}: {count}")

            # -------- GRAPH --------
            st.session_state.data.append(len(detected_classes))
            df = pd.DataFrame(st.session_state.data, columns=["Objects"])
            st.line_chart(df)

            # -------- ALERT SYSTEM --------
            st.markdown("## 🚨 AI Alerts")

            if mode == "Industrial Safety":
                if "person" in detected_classes and "helmet" not in detected_classes:
                    st.error("⚠️ Safety Violation: No Helmet!")
                else:
                    st.success("✅ Safe")

            elif mode == "Traffic Monitoring":
                if counts.get("car", 0) > 3:
                    st.error("🚗 Heavy Traffic Detected!")
                elif counts.get("car", 0) > 1:
                    st.warning("⚠️ Moderate Traffic")
                else:
                    st.success("✅ Smooth Traffic")

            else:
                st.info("ℹ️ General Detection Mode")

            # -------- DOWNLOAD --------
            output_path = "output.jpg"
            cv2.imwrite(output_path, res_plotted)

            with open(output_path, "rb") as file:
                st.download_button("📥 Download Result", file, "result.jpg")

            # -------- HISTORY --------
            st.session_state.history.append({
                "file": uploaded_file.name,
                "objects": detected_classes
            })

            st.markdown("---")

# =========================================================
# ================= WEBCAM MODE ============================
# =========================================================
elif input_mode == "Webcam":

    st.markdown("## 🎥 Live Webcam Detection")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()

        if not ret:
            st.warning("Camera not working")
            break

        # -------- TRACKING --------
        results = model.track(frame, conf=confidence, persist=True)
        res_plotted = results[0].plot()

        FRAME_WINDOW.image(res_plotted)

        # -------- CLASSES --------
        detected_classes = [
            model.names[int(box.cls[0])]
            for box in results[0].boxes
        ]

        counts = Counter(detected_classes)

        # -------- GRAPH --------
        st.session_state.data.append(len(detected_classes))
        df = pd.DataFrame(st.session_state.data, columns=["Objects"])
        st.line_chart(df)

        # -------- ALERT --------
        if mode == "Industrial Safety":
            if "person" in detected_classes and "helmet" not in detected_classes:
                st.error("⚠️ No Helmet Detected!")

        elif mode == "Traffic Monitoring":
            if counts.get("car", 0) > 3:
                st.error("🚗 Heavy Traffic!")
            elif counts.get("car", 0) > 1:
                st.warning("⚠️ Moderate Traffic")

    cap.release()

# =========================================================
# ================= HISTORY ================================
# =========================================================
st.markdown("## 🕓 Detection History")

for item in st.session_state.history[-5:]:
    st.write(f"📂 {item['file']} → {item['objects']}")
