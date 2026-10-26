import cv2
import numpy as np

def align_to_reference(reference, inspected):
    h, w = reference.shape[:2]
    resized = cv2.resize(inspected, (w, h))
    try:
        a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1500)
        kp1, des1 = orb.detectAndCompute(a, None)
        kp2, des2 = orb.detectAndCompute(b, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return resized
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(des2, des1, k=2)
        good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.72*n.distance]
        if len(good) < 8:
            return resized
        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if matrix is None or inliers is None or int(inliers.sum()) < 6:
            return resized
        return cv2.warpPerspective(resized, matrix, (w, h))
    except cv2.error:
        return resized

def inspect_against_reference(reference, inspected, threshold=25, min_area=180):
    if reference is None or inspected is None:
        raise ValueError("Both reference and inspected images are required.")
    if reference.ndim != 3 or inspected.ndim != 3:
        raise ValueError("Expected color images in BGR format.")
    aligned = align_to_reference(reference, inspected)
    a = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    diff = cv2.absdiff(a, b)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated, regions = aligned.copy(), []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < int(min_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": area})
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(annotated, f"Difference {len(regions)}", (x, max(20, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    regions.sort(key=lambda r: r["area"], reverse=True)
    return {"annotated": annotated, "regions": regions, "difference_mask": mask, "aligned": aligned}

def run_yolo_detection(image, weights_path, confidence=0.25):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    results = model.predict(source=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), conf=confidence, verbose=False)
    rendered, rows = image.copy(), []
    for result in results:
        for box in result.boxes or []:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().tolist()]
            class_id = int(box.cls[0].cpu().item())
            score = float(box.conf[0].cpu().item())
            label = f"{result.names.get(class_id, class_id)} {score:.2f}"
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 180, 0), 2)
            cv2.putText(rendered, label, (x1, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
            rows.append({"class": result.names.get(class_id, str(class_id)), "confidence": round(score, 4),
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return rows, rendered
