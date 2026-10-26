# PCB Placement Inspector

A student capstone prototype for AI-assisted PCB assembly inspection. It compares an inspected board against a known-good reference image and optionally runs a trained YOLO object detector.

## Features
- OpenCV image alignment and reference comparison
- Highlights regions that differ from the reference
- Optional YOLO inference for labelled, visible defects
- Streamlit upload-and-inspect interface
- YOLO training and evaluation scripts
- Editable report and presentation starter

> **Honest project status:** The supplied sample images are synthetic illustrations for testing the interface. No trained weights, real-world test results, or accuracy claims are included. Train and evaluate with a labelled dataset of real PCB images before claiming model performance.

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/generate_sample_data.py
streamlit run app.py
```
Click **Load included synthetic example** or upload a known-good reference and an inspection image.

## Train YOLO on your own data
1. Collect representative real PCB images with normal boards and defects.
2. Annotate visible defect regions with CVAT or LabelImg.
3. Split by physical board or capture session to reduce data leakage.
4. Arrange files as described in `data/README.md`.
5. Edit paths and class names in `data/pcb_defects.yaml`.
6. Run:

```bash
python src/train.py --data data/pcb_defects.yaml --epochs 80 --imgsz 640
python src/evaluate.py --weights runs/detect/pcb_inspection/weights/best.pt --data data/pcb_defects.yaml
```

Copy the trained weights to `models/best.pt` to enable the optional detector in the app.

## Dataset layout
```text
your_dataset/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```
YOLO labels use `class_id x_center y_center width height`, normalized to 0–1. Empty label files are valid for images with no visible defect. Decide a consistent policy for missing components: a detector cannot see an absent object, so reference comparison or footprint-based checks may be more appropriate.

## Project structure
- `app.py` — Streamlit interface
- `src/reference_inspection.py` — image alignment, difference detection, optional YOLO inference
- `src/train.py` — training entry point
- `src/evaluate.py` — held-out evaluation
- `src/generate_sample_data.py` — synthetic UI examples
- `docs/PROJECT_REPORT.md` — editable report draft
- `docs/PCB_Placement_Error_Detection_Presentation.pptx` — editable presentation starter

## Limitations
Lighting, glare, camera movement, board revisions, and alignment can create false positives or hide defects. Treat flagged areas as candidates for review, not confirmed defects. This prototype is not a certified manufacturing inspection system.
