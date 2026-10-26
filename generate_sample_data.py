from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

def make_board(missing=False, shifted=False):
    img = np.full((520, 760, 3), (42, 92, 48), dtype=np.uint8)
    cv2.rectangle(img, (28, 28), (732, 492), (80, 150, 90), 3)
    for y in [100, 180, 260, 340, 420]:
        cv2.line(img, (70, y), (690, y), (55, 125, 75), 2)
    for x in [120, 240, 360, 480, 600]:
        cv2.line(img, (x, 60), (x, 460), (55, 125, 75), 2)
    for x, y in [(52, 52), (708, 52), (52, 468), (708, 468)]:
        cv2.circle(img, (x, y), 13, (170, 170, 160), -1)
        cv2.circle(img, (x, y), 6, (35, 35, 35), -1)
    positions = [(150,130),(300,130),(450,130),(600,130),(150,250),(300,250),
                 (450,250),(600,250),(150,370),(300,370),(450,370),(600,370)]
    for i, (x, y) in enumerate(positions):
        if missing and i == 5:
            continue
        if shifted and i == 8:
            x, y = x+28, y+12
        w, h = (52, 28) if i % 3 else (30, 48)
        cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (28,28,30), -1)
        cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (190,190,185), 2)
        for px in range(x-w//2+4, x+w//2, 8):
            cv2.line(img, (px,y-h//2-5), (px,y-h//2), (210,185,110), 2)
            cv2.line(img, (px,y+h//2), (px,y+h//2+5), (210,185,110), 2)
        cv2.putText(img, f"U{i+1}", (x-w//2,y+h//2+17), cv2.FONT_HERSHEY_SIMPLEX, .35, (225,230,220), 1)
    cv2.putText(img, "SYNTHETIC DEMO - NOT A REAL PCB", (125,30), cv2.FONT_HERSHEY_SIMPLEX, .55, (245,245,245), 1)
    return img

if __name__ == "__main__":
    cv2.imwrite(str(OUT / "reference_board.png"), make_board())
    cv2.imwrite(str(OUT / "example_board_with_change.png"), make_board(missing=True, shifted=True))
    print(f"Synthetic examples saved to {OUT}")
