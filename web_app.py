import streamlit as st
import cv2
import os
import tempfile
import numpy as np

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Parking System",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Real-Time Smart Parking System")

st.write(
    "Parking Slot Occupancy Detection Using "
    "YOLO and Computer Vision"
)


# ==========================================
# 2. MODEL PATH
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_CANDIDATES = [
    SCRIPT_DIR,
    SCRIPT_DIR / "Smart-Parking-System-main",
    SCRIPT_DIR.parent / "Smart-Parking-System-main"
]

MODEL_PATH = None

for folder in PROJECT_CANDIDATES:

    possible_models = [
        folder / "V6 best.pt",
        folder / "Models" / "V6 best.pt",
        folder / "models" / "parking_detection" / "V6 best.pt"
    ]

    for model_file in possible_models:

        if model_file.exists():
            MODEL_PATH = model_file
            break

    if MODEL_PATH:
        break


if MODEL_PATH is None:

    st.error("YOLO Model Not Found!")

    st.write("Please upload V6 best.pt to your GitHub repository.")

    st.code("""
V6 best.pt
Models/V6 best.pt
models/parking_detection/V6 best.pt
""")

    st.stop()


st.success("YOLO Model Found Successfully!")


# ==========================================
# 3. LOAD YOLO MODEL
# ==========================================

@st.cache_resource
def load_model(path):

    return YOLO(str(path))


try:

    model = load_model(str(MODEL_PATH))

    st.success("YOLO Model Loaded Successfully!")

except Exception as e:

    st.error(f"Model Loading Error: {e}")

    st.stop()


# ==========================================
# 4. PARKING DETECTION FUNCTION
# ==========================================

def detect_parking(frame):

    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    occupied = 0
    vacant = 0

    if result.boxes is not None:

        for cls in result.boxes.cls.tolist():

            class_name = str(
                model.names[int(cls)]
            ).lower()

            if class_name in ["car", "occupied"]:
                occupied += 1

            elif class_name in ["free", "vacant"]:
                vacant += 1

    output = result.plot()

    return output, occupied, vacant


# ==========================================
# 5. LIVE CAMERA PROCESSOR
# ==========================================

class ParkingCamera(VideoProcessorBase):

    def __init__(self):

        self.model = model

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = self.model.predict(
            source=image,
            conf=0.5,
            verbose=False
        )

        result = results[0]

        occupied = 0
        vacant = 0

        if result.boxes is not None:

            for cls in result.boxes.cls.tolist():

                class_name = str(
                    self.model.names[int(cls)]
                ).lower()

                if class_name in ["car", "occupied"]:
                    occupied += 1

                elif class_name in ["free", "vacant"]:
                    vacant += 1

        output = result.plot()

        cv2.putText(
            output,
            f"Occupied: {occupied}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            output,
            f"Vacant: {vacant}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Total: {occupied + vacant}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        return type(frame).from_ndarray(
            output,
            format="bgr24"
        )


# ==========================================
# 6. DASHBOARD TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([
    "📷 Image Upload",
    "🎬 Video Upload",
    "📹 Live Camera"
])


# ==========================================
# 7. IMAGE UPLOAD
# ==========================================

with tab1:

    st.header("📷 Parking Image Detection")

    image_file = st.file_uploader(
        "Upload Parking Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if image_file:

        image = Image.open(image_file).convert("RGB")

        image_array = np.array(image)

        st.image(
            image,
            caption="Uploaded Parking Image",
            width="stretch"
        )

        if st.button("🔍 Detect Parking", key="image_button"):

            with st.spinner("Detecting Parking Slots..."):

                frame = cv2.cvtColor(
                    image_array,
                    cv2.COLOR_RGB2BGR
                )

                output, occupied, vacant = detect_parking(frame)

                st.subheader("Detection Result")

                st.image(
                    cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                    caption="YOLO Detection Output",
                    width="stretch"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Occupied Slots", occupied)
                col2.metric("Vacant Slots", vacant)
                col3.metric("Total Slots", occupied + vacant)


# ==========================================
# 8. VIDEO UPLOAD
# ==========================================

with tab2:

    st.header("🎬 Parking Video Detection")

    video_file = st.file_uploader(
        "Upload Parking Video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if video_file:

        video_bytes = video_file.getvalue()

        st.video(video_bytes)

        if st.button("▶ Run Video Detection", key="video_button"):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(video_file.name).suffix
                ) as temp_input:

                    temp_input.write(video_bytes)
                    input_path = temp_input.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():

                    st.error("Unable to open uploaded video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)

                if not fps or fps <= 0:
                    fps = 20

                width = int(
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                )

                height = int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp_output:

                    output_path = temp_output.name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():

                    st.error("Unable to create output video.")
                    st.stop()

                progress = st.progress(0)
                status = st.empty()

                frame_count = 0
                occupied = 0
                vacant = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    output, occupied, vacant = detect_parking(frame)

                    writer.write(output)

                    frame_count += 1

                    if total_frames > 0:

                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                    status.write(
                        f"Processing Frame: {frame_count}/{total_frames}"
                    )

                cap.release()
                cap = None

                writer.release()
                writer = None

                st.success("Video Detection Completed Successfully!")

                st.subheader("Processed Video")

                with open(output_path, "rb") as output_file:

                    processed_video = output_file.read()

                st.video(processed_video)

                col1, col2, col3 = st.columns(3)

                col1.metric("Occupied Slots", occupied)
                col2.metric("Vacant Slots", vacant)
                col3.metric("Total Slots", occupied + vacant)

                st.caption(
                    "Counts above represent the final processed frame."
                )

            except Exception as e:

                st.error(f"Video Processing Error: {e}")

            finally:

                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

                if output_path and os.path.exists(output_path):
                    os.remove(output_path)


# ==========================================
# 9. LIVE CAMERA
# ==========================================

with tab3:

    st.header("📹 Live Parking Detection")

    st.info(
        "Click START to activate your device camera. "
        "Allow camera permission in your browser."
    )

    webrtc_streamer(
        key="smart-parking-camera",
        video_processor_factory=ParkingCamera,

        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },

        async_processing=True
    )

    st.caption(
        "Occupied, Vacant and Total counts "
        "are displayed directly on the live video."
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Parking System | Python | YOLO | "
    "OpenCV | Streamlit"
)
