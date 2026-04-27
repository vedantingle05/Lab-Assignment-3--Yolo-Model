from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
model = YOLO(r"runs/detect/runs/detect/detection_run/weights/best.pt")

@app.route("/", methods=["GET"])
def home():
    return "<h2>YOLO Detection API Running!</h2><p>POST image to /predict</p>"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image. Use key 'image'"}), 400
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    results = model.predict(img, conf=0.25)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class":      model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox":       [round(x, 1) for x in box.xyxy[0].tolist()]
            })
    return jsonify({"total": len(detections), "detections": detections})

if __name__ == "__main__":
    print("API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)