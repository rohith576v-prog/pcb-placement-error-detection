import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reference_inspection import inspect_against_reference

def test_identical_images():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = inspect_against_reference(image, image.copy(), threshold=25, min_area=20)
    assert result["annotated"].shape == image.shape
    assert result["regions"] == []

def test_changed_patch_is_flagged():
    reference = np.zeros((120, 120, 3), dtype=np.uint8)
    inspected = reference.copy()
    cv2.rectangle(inspected, (40, 40), (75, 75), (255, 255, 255), -1)
    result = inspect_against_reference(reference, inspected, threshold=10, min_area=20)
    assert len(result["regions"]) >= 1
