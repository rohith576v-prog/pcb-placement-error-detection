# AI-Based Component Placement Error Detection in PCB Assembly

**Student:** [Your name]  
**Department:** Electronics and Communication Engineering  
**Institution:** [Institution]  
**Project guide:** [Guide name]  
**Date:** [Submission date]

## Abstract
Printed Circuit Board assembly requires components to be present and positioned correctly. This project explores a computer-vision workflow that compares an inspected PCB image with a known-good reference and optionally applies a YOLO object detector to localize visible defects. The application highlights image regions for human review. Reliable performance claims require a representative labelled dataset of real PCB images.

## Problem statement
Build a prototype that flags possible missing, shifted, incorrectly oriented, or wrong components and visualizes suspicious regions.

## Objectives
- Prepare and document a labelled PCB dataset.
- Implement image preprocessing and reference comparison.
- Train an object detector on annotated defects.
- Evaluate using precision, recall, F1-score and mAP where applicable.
- Present detections through a simple inspection interface.

## Methodology
1. Capture images under repeatable lighting and camera conditions.
2. Annotate defects using a consistent policy.
3. Split by physical board or capture session into train, validation and test sets.
4. Preprocess images and apply realistic, label-preserving augmentation.
5. Align the inspected image to a known-good reference and compare image regions.
6. Fine-tune a pretrained YOLO model for visible defect localization.
7. Evaluate on held-out data and inspect false positives and false negatives.
8. Present annotated output in the Streamlit interface.

## System design
The pipeline consists of image input, preprocessing/alignment, reference comparison, optional YOLO inference, visualization and a results summary. Reference comparison is useful for missing-component checks because the expected part may be absent from the inspected image.

## Implementation
- `app.py`: user interface.
- `src/reference_inspection.py`: alignment, difference mask, contour extraction and optional YOLO inference.
- `src/train.py`: model training entry point.
- `src/evaluate.py`: evaluation wrapper.
- `src/generate_sample_data.py`: synthetic images for testing the UI.

The supplied sample images are synthetic illustrations and are not evidence of real-world performance.

## Evaluation
Report the number of boards/images in each split, per-class precision and recall, F1-score, mAP@50, mAP@50–95, confusion matrix, example false positives/negatives, and inference time with hardware details.

| Metric | Held-out test result |
|---|---:|
| Precision | [measure] |
| Recall | [measure] |
| F1-score | [measure] |
| mAP@50 | [measure] |
| mAP@50–95 | [measure] |
| Inference time | [measure] |

Do not fill this table with guessed results.

## Limitations
Pixel differences are sensitive to lighting, reflections, registration errors and board revisions. YOLO only learns the classes represented by sufficient labelled examples. Missing components require a reference-aware approach or footprint-based checks. This prototype is not a certified manufacturing inspection system and should not be the sole basis for production acceptance.

## Future work
- Use Gerber/CAD data or component coordinates to define expected footprints.
- Improve registration and illumination normalization.
- Add polarity/orientation checks and component identification.
- Collect data across multiple boards, cameras and production lots.
- Measure false-alarm rates and inspection latency.

## Conclusion
This repository provides a modular starting point for PCB placement inspection using reference-image comparison and optional object detection. Robust detection and performance claims depend on training and evaluating with suitable real-world data.
