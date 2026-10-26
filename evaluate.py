import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained weights on a held-out split.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/pcb_defects.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, plots=True)
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
