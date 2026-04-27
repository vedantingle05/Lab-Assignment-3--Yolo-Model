# ============================================================
#   YOLO Lab Assignment 3 — FIXED Complete Code
# ============================================================

from ultralytics import YOLO
import os

# ============================================================
# HELPER — finds the actual saved model path automatically
# ============================================================

def find_best_model(base_task, run_name):
    """Searches for best.pt regardless of nesting"""
    possible = [
        f"runs/{base_task}/{run_name}/weights/best.pt",
        f"runs/{base_task}/runs/{base_task}/{run_name}/weights/best.pt",
    ]
    for p in possible:
        if os.path.exists(p):
            return p
    # search recursively as fallback
    for root, dirs, files in os.walk("runs"):
        for f in files:
            if f == "best.pt" and run_name in root:
                return os.path.join(root, f)
    return None


# ============================================================
# PART A — Object Detection
# ============================================================

def train_detection():
    print("\n" + "="*50)
    print("PART A — Object Detection Training")
    print("="*50)

    model = YOLO("yolov8n.pt")
    model.train(
        data="coco8.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu",
        name="detection_run",
        project="runs/detect",
        workers=0,
        exist_ok=True
    )

    best = find_best_model("detect", "detection_run")
    print(f"\nDetection Training Done! Model at: {best}")

    print("\nRunning inference...")
    model2 = YOLO(best)
    model2.predict(
        source="https://ultralytics.com/images/bus.jpg",
        conf=0.25,
        save=True,
        project="runs/detect",
        name="detection_inference",
        exist_ok=True
    )
    print("Inference done. Check runs/detect/detection_inference/")
    return best


# ============================================================
# PART B — Classification
# ============================================================

def train_classification():
    print("\n" + "="*50)
    print("PART B — Classification Training")
    print("="*50)

    model = YOLO("yolov8n-cls.pt")
    model.train(
        data="mnist160",
        epochs=20,
        imgsz=64,
        batch=32,
        device="cpu",
        name="classification_run",
        project="runs/classify",
        workers=0,
        exist_ok=True
    )

    best = find_best_model("classify", "classification_run")
    print(f"\nClassification Training Done! Model at: {best}")

    print("\nRunning inference...")
    model2 = YOLO(best)
    model2.predict(
        source="https://ultralytics.com/images/bus.jpg",
        save=True,
        project="runs/classify",
        name="classification_inference",
        exist_ok=True
    )
    print("Inference done. Check runs/classify/classification_inference/")
    return best


# ============================================================
# PART C — Pose Estimation
# ============================================================

def train_pose():
    print("\n" + "="*50)
    print("PART C — Pose Estimation Training")
    print("="*50)

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data="coco8-pose.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device="cpu",
        name="pose_run",
        project="runs/pose",
        workers=0,
        exist_ok=True
    )

    best = find_best_model("pose", "pose_run")
    print(f"\nPose Training Done! Model at: {best}")

    print("\nRunning inference...")
    model2 = YOLO(best)
    model2.predict(
        source="https://ultralytics.com/images/bus.jpg",
        conf=0.25,
        save=True,
        project="runs/pose",
        name="pose_inference",
        exist_ok=True
    )
    print("Inference done. Check runs/pose/pose_inference/")
    return best


# ============================================================
# PART D — OBB
# ============================================================

def train_obb():
    print("\n" + "="*50)
    print("PART D — OBB Training")
    print("="*50)

    model = YOLO("yolov8n-obb.pt")
    model.train(
        data="dota8.yaml",
        epochs=20,
        imgsz=640,
        batch=8,
        device="cpu",
        name="obb_run",
        project="runs/obb",
        workers=0,
        exist_ok=True
    )

    best = find_best_model("obb", "obb_run")
    print(f"\nOBB Training Done! Model at: {best}")

    print("\nRunning inference...")
    model2 = YOLO(best)
    model2.predict(
        source="https://ultralytics.com/images/boats.jpg",
        conf=0.25,
        save=True,
        project="runs/obb",
        name="obb_inference",
        exist_ok=True
    )
    print("Inference done. Check runs/obb/obb_inference/")
    return best


# ============================================================
# SAVE FLASK APP
# ============================================================

def save_flask_app(detect_model_path):
    code = f"""
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO(r"{detect_model_path}")

@app.route("/", methods=["GET"])
def home():
    return "<h2>YOLO Detection API Running!</h2><p>POST image to /predict</p>"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({{"error": "No image. Use key 'image'"}}), 400
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    results = model.predict(img, conf=0.25)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({{
                "class":      model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox":       [round(x, 1) for x in box.xyxy[0].tolist()]
            }})
    return jsonify({{"total": len(detections), "detections": detections}})

if __name__ == "__main__":
    print("API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
""".strip()

    with open("app.py", "w") as f:
        f.write(code)
    print("\napp.py saved. Run with: python app.py")


# ============================================================
# PRINT FINAL SUMMARY WITH REAL METRICS
# ============================================================

def print_summary():
    print("\n" + "="*60)
    print("FINAL SUMMARY — Copy these values into your Word report")
    print("="*60)

    tasks = [
        ("A - Detection",      "detect",   "detection_run"),
        ("B - Classification", "classify", "classification_run"),
        ("C - Pose",           "pose",     "pose_run"),
        ("D - OBB",            "obb",      "obb_run"),
    ]

    for name, task, run in tasks:
        best = find_best_model(task, run)
        if best:
            # load results csv
            csv_path = best.replace("weights/best.pt", "results.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    lines = f.readlines()
                header = [h.strip() for h in lines[0].split(",")]
                # get best epoch row (last line)
                last = [v.strip() for v in lines[-1].split(",")]
                print(f"\n Part {name}")
                print(f"   Model path : {best}")
                for h, v in zip(header, last):
                    if any(k in h for k in ["mAP","loss","acc","precision","recall"]):
                        print(f"   {h:30s}: {v}")
            else:
                print(f"\n Part {name}: model found at {best} (no CSV)")
        else:
            print(f"\n Part {name}: NOT FOUND — training may have failed")

    print("\n" + "="*60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("YOLO Lab Assignment 3 — All 4 Parts")
    print("CPU-only mode. Total time: ~1-2 hours\n")

    detect_path  = train_detection()
    train_classification()
    train_pose()
    train_obb()

    save_flask_app(detect_path)
    print_summary()

    print("\nALL DONE!")
    print("Now run:  python app.py  to start deployment server")