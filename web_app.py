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

    st.write("Please check that V6 best.pt is uploaded to GitHub.")

    st.write("Expected locations:")
    st.code("""
V6 best.pt
Models/V6 best.pt
models/parking_detection/V6 best.pt
""")

    st.stop()


st.success("YOLO Model Found Successfully!")
st.caption(f"Model Path: {MODEL_PATH}")
