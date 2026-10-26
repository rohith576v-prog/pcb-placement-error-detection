from pathlib import Path
import sys
import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from reference_inspection import inspect_against_reference, run_yolo_detection

st.set_page_config(page_title="PCB Placement Inspector", page_icon="🔎", layout="wide")
st.title("PCB Placement Inspector")
st.caption("A prototype for AI-assisted visual inspection of PCB component placement.")

with st.sidebar:
    st.header("Inspection settings")
    threshold = st.slider("Difference sensitivity", 5, 80, 25)
    min_area = st.slider("Minimum changed region (pixels)", 20, 2000, 180, step=20)
    use_yolo = st.checkbox("Run optional YOLO model", value=False)
    model_path = st.text_input("YOLO weights path", "models/best.pt")
    st.info("Consistent camera position and lighting improve reference comparison.")

def decode(upload):
    if upload is None:
        return None
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Known-good reference")
    ref_file = st.file_uploader("Reference image", type=["png", "jpg", "jpeg"], key="ref")
with c2:
    st.subheader("Board to inspect")
    test_file = st.file_uploader("Inspection image", type=["png", "jpg", "jpeg"], key="test")

if st.button("Load included synthetic example"):
    st.session_state["sample"] = True

if st.session_state.get("sample"):
    ref = cv2.imread(str(ROOT / "data/sample/reference_board.png"))
    test = cv2.imread(str(ROOT / "data/sample/example_board_with_change.png"))
else:
    ref, test = decode(ref_file), decode(test_file)

if ref is not None and test is not None:
    result = inspect_against_reference(ref, test, threshold=threshold, min_area=min_area)
    a, b = st.columns(2)
    with a:
        st.image(cv2.cvtColor(test, cv2.COLOR_BGR2RGB), caption="Inspection image", use_container_width=True)
    with b:
        st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), caption="Highlighted differences", use_container_width=True)
    st.subheader("Reference comparison")
    st.metric("Changed regions flagged", len(result["regions"]))
    if result["regions"]:
        st.warning("Differences were found. Review them; an image difference is not automatically a confirmed defect.")
        st.dataframe([{"Region": i+1, **r} for i, r in enumerate(result["regions"])], use_container_width=True)
    else:
        st.success("No regions passed the current threshold. This does not prove the board is defect-free.")
    if use_yolo:
        weights = ROOT / model_path
        if weights.exists():
            try:
                detections, rendered = run_yolo_detection(test, str(weights))
                st.subheader("YOLO detections")
                st.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.write(detections if detections else "No detections above the model threshold.")
            except Exception as exc:
                st.error(f"YOLO inference failed: {exc}")
        else:
            st.error(f"Weights not found: {weights}. Train a model and place the weights there first.")
    ok, encoded = cv2.imencode(".png", result["annotated"])
    if ok:
        st.download_button("Download annotated image", encoded.tobytes(), "pcb_inspection_result.png", "image/png")
else:
    st.markdown("### How to use")
    st.write("Upload a known-good PCB image and the board to inspect, or click the synthetic example button.")
    st.write("Adjust sensitivity, review every highlighted region, and download the annotated result.")
st.caption("Student capstone prototype. Validate on real labelled data before making performance claims.")
