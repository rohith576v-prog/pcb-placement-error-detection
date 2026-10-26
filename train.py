import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on labelled PCB defect images.")
    parser.add_argument("--data", default="data/pcb_defects.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project="runs/detect", name="pcb_inspection", patience=15, seed=42)

if __name__ == "__main__":
    main()
